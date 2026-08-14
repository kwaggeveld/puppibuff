from .codec import Codec
from ..datasets import Dataset

import numpy as np

from numpy.typing import NDArray

#-----------------------------------------------------------------------------

class FixedMCodec(Codec):
    """Per-channel codec for fixed multiplicity M (pt, eta, phi) events.

    pt  -> log1p -> normalise (over all jets)
    eta -> normalise          (over all jets)
    phi -> (sin_phi, cos_phi), or plain normalise if `s1phi = False`
    """

    s_EXPORT_KEYS = [ channel + "_" + attr
                      for channel in ( "pt", "eta", "phi" )
                      for attr    in ( "mean", "std", "min", "max" )] \
                    + [ "s1phi", "n_features", "multiplicity" ]

    s_DECODED = [ "pt", "eta", "phi" ]  # What `decode` returns, per slot

    s_FRACTION_BITS = 12                # Decoded outputs' fractional precision

    s_EXPM1_FRAC_BITS = 8               # `expm1` table entries per unit of
                                        # log1p(pt). A bin's width in that space
                                        # is the relative error in pt, so this
                                        # is 0.39% -- just under the BDTs' own
                                        # ap_fixed truncation bias (0.4 - 1.1%).


    def check_dataset(self, data: Dataset) -> None:
        super().check_dataset(data)     # Asserts type

        ref_shape= None                
        for channel in data.channels(): # Check each channel's shape: should be
            arr = data[channel]         #   (N, M) = (num_events, multiplicity)
                                        # for each channel.
            if arr.dtype == object:
                raise ValueError(
                    f"Channel {channel!r} is ragged. "
                    f"FixedMCodec requires fixed jet multiplicity."
                )
            if arr.ndim not in (1, 2):
                raise ValueError(
                    f"channel {channel!r} must be flattened 1D (N,) "
                    f"or 2D (N, M), got shape {arr.shape}"
                )
            if ref_shape is None:
                ref_shape = arr.shape
            elif arr.shape != ref_shape:
                raise ValueError(
                    f"Channel {channel!r} has shape {arr.shape}, "
                    f"expected {ref_shape} to match other channels"
                )


    def fit(self, data: Dataset) -> None:
        self.check_dataset(data)

        self._fit_stats(data["pt"], data["eta"], data["phi"])

        self.n_features   = len(data.channels()) + self.s1phi   # phi -> (sin, cos) adds one
        self.multiplicity = data["pt"].shape[1] if data["pt"].ndim == 2 else 1

    def encode(self, data: Dataset) -> NDArray:
        self.check_dataset(data)

        encoded_channels = self._encode_channels(
            data["pt"], data["eta"], data["phi"]
        )
                                        # from 4 x (N, M) to (N, 4, M)
        return np.stack([*encoded_channels], axis = 1)


    def decode(self, out: NDArray) -> dict[str, NDArray]:
        encoded_channels = np.moveaxis(out, 1, 0)
        return self._decode_channels(*encoded_channels)


    def group_sizes(self) -> list[int]:
        """One block of size `M` for each feature"""
        return [self.multiplicity] * self.n_features


    @property
    def n_decoded(self) -> int:
        """Every decoded channel, for every slot."""
        return len(self.s_DECODED) * self.multiplicity


    @property
    def decoded_precision(self) -> str:
        """An `ap_fixed` covering every decoded channel's observed range."""
        largest = max(abs(self.pt_min),  abs(self.pt_max),
                      abs(self.eta_min), abs(self.eta_max),
                      np.pi)            # phi wraps to [-pi, pi)

        integer = int(np.ceil(np.log2(largest))) + 1        # Plus a sign bit

        return f"ap_fixed<{ integer + self.s_FRACTION_BITS }, { integer }>"


    @property
    def expm1_scaling(self) -> int:
        """`expm1` table entries per unit of log1p(pt). Used to convert value 
        <-> table idx. idx / scale in the init, value * scale in the lookup.
        """
        return 1 << self.s_EXPM1_FRAC_BITS


    @property
    def expm1_table_size(self) -> int:
        """Enough `expm1` entries to reach log1p(pt_max) given the scaling,
        so that clamping the index saturates at the observed range.
        """
        return int(np.ceil(np.log1p(self.pt_max) * self.expm1_scaling))


    def decode_cpp(self) -> str:
        if self.s1phi:                  # `_decode_channels` reaches for arctan2
            raise NotImplementedError(
                "`decode_cpp` cannot emit the s1phi (sin, cos) decoding of phi. "
                "Fit the codec with s1phi = False."
            )

        return f"""#include "flowhls.h"

#ifdef __SYNTHESIS__                    // Defined by Vitis HLS when synthesising:
    #include "hls_math.h"               // https://docs.amd.com/r/en-US/ug1399-vitis-hls/System-Calls
#else                                   // but normally `hls_math` is unavailable
    #include <cmath>                    // so fallback
    namespace hls = std;
#endif
                                        // Slots per event
static size_t const multiplicity = { self.multiplicity };

                                        // Fitted statistics. 
static accum_t   const pt_mean  = { self.pt_mean };
static accum_t   const pt_std   = { self.pt_std };
static decoded_t const eta_mean = { self.eta_mean };
static decoded_t const eta_std  = { self.eta_std };
                                        // phi's, in turns rather than radians
static accum_t   const phi_mean = { self.phi_mean / (2 * np.pi) };
static accum_t   const phi_std  = { self.phi_std / (2 * np.pi) };

                                        // Observed ranges.
static float     const pt_min  = { self.pt_min };   // Read only by table fill -> float
static decoded_t const eta_min = { self.eta_min };
static decoded_t const eta_max = { self.eta_max };

static decoded_t const two_pi = { 2 * np.pi };

// Lookup table design adapted from: 
// https://github.com/fastmachinelearning/hls4ml/blob/main/hls4ml/templates/vivado/nnet_utils/nnet_activation.h
static size_t const expm1_scaling    = 256;
static size_t const expm1_table_size = 1783;

void init_expm1_table(decoded_t table[expm1_table_size])
{{                                       // Fill table idx-by-idx
    for (size_t idx = 0; idx != expm1_table_size; ++idx)
    {{                                   // Convert idx -> X-value
        float in_val = idx / float{{ expm1_scaling }};
                                        // Clip minimum table entry to pt_min
        table[idx] = hls::fmax(hls::expm1(in_val), pt_min);
    }}
}}

inline decoded_t expm1_lookup(accum_t value)
{{
    #pragma HLS inline
#ifdef __HLS_SYN__
    bool initialised = false;
    decoded_t expm1_table[expm1_table_size];
#else                                   // Make the table static if emulating, so
    static bool initialised = false;    // that it's shared over multiple calls
    static decoded_t expm1_table[expm1_table_size];
#endif
    if (!initialised)
    {{
        init_expm1_table(expm1_table);
        initialised = true;
    }}

    int table_index = value * expm1_scaling;

    if (table_index < 0)
        table_index = 0;
    else if (table_index > int(expm1_table_size) - 1)
        table_index = expm1_table_size - 1;

    return expm1_table[table_index];
}}

inline decoded_t clip(decoded_t value, decoded_t low, decoded_t high)
{{
    #pragma HLS inline
    return value < low  ? low  :
           value > high ? high : value;
}}

                                        // `phi` is in unit of turns, not radians. 
                                        // With zero integer bits we have a range
                                        // of [-0.5, 0.5]. We can then use overflow 
                                        // acting as wrapping.
using turns_t = ap_fixed<{ self.s_FRACTION_BITS + 4 }, 0>;  // 4 > log2(2pi)

inline decoded_t wrap_phi(accum_t value)
{{
    #pragma HLS inline                  // Narrowing wraps back to [-0.5, 0.5]
    turns_t const turns = value * phi_std + phi_mean;
    return turns * two_pi;
}}

void { self.s_DECODE_TOP }(accum_arr_t x, decoded_arr_t decoded)
{{
    #pragma HLS pipeline
    #pragma HLS array_partition variable=x
    #pragma HLS array_partition variable=decoded

    for (size_t idx = 0; idx != multiplicity; ++idx)
    {{
        #pragma HLS unroll
        decoded_t const pt  = expm1_lookup(x[idx] * pt_std + pt_mean);
        decoded_t const eta = x[multiplicity + idx] * eta_std + eta_mean;
        decoded_t const phi = wrap_phi(x[2 * multiplicity + idx]);

        decoded[idx]                    = pt;           // The LUT clips
        decoded[multiplicity + idx]     = clip(eta, eta_min, eta_max);
        decoded[2 * multiplicity + idx] = phi;          // `wrap_phi` rescales
    }}
}}

"""


    def _fit_stats(self, pt: NDArray, eta: NDArray, phi: NDArray) -> None:
        """Set pt/eta/phi mean/std/min/max from arrays."""
        logpt = np.log1p(pt)

        self.pt_mean  = float(logpt.mean())
        self.pt_std   = float(logpt.std())
        self.eta_mean = float(eta.mean())
        self.eta_std  = float(eta.std())
        self.phi_mean = float(phi.mean())       # Unused when self.s1phi == True
        self.phi_std  = float(phi.std())
                                                
        self.pt_min  = float(np.nanmin(pt))     # Observed physical ranges 
        self.pt_max  = float(np.nanmax(pt))     # to clip samples to
        self.eta_min = float(np.nanmin(eta))
        self.eta_max = float(np.nanmax(eta))
        self.phi_min = float(np.nanmin(phi))
        self.phi_max = float(np.nanmax(phi))


    def _encode_channels(
        self, pt: NDArray, eta: NDArray, phi: NDArray
    ) -> tuple[NDArray, ...]:
        """Normalise log(1 + pt) and eta. Encode phi as (sin(phi), cos(phi)) 
        if self.s1phi, else normalise.
        """
        std_pt  = (np.log1p(pt) - self.pt_mean) / self.pt_std
        std_eta = (eta - self.eta_mean) / self.eta_std

        if self.s1phi:
            return std_pt, std_eta, np.sin(phi), np.cos(phi)

        std_phi = (phi - self.phi_mean) / self.phi_std
        return std_pt, std_eta, std_phi


    def _decode_channels(                      # (sin, cos) or std_phi
        self, std_pt: NDArray, std_eta: NDArray, *phi_channels: NDArray
    ) -> dict[str, NDArray]:
        """Rescale e^pt - 1 and eta back to observed mean, std. Decode phi
        as arctan2(sin(phi), cos(phi)) if self.s1phi, else rescale back, then
        wrap to [-pi, pi).
        """
        pt  = np.expm1(std_pt * self.pt_std + self.pt_mean)
        eta = std_eta * self.eta_std + self.eta_mean

                                        # if self.s1phi: 
                                        #   phi_channels = (sin_phi, cos_phi)
                                        # else:     phi_channels = (std_phi,)
        phi = (np.arctan2(*phi_channels) if self.s1phi          # -> (-pi, pi]
                else phi_channels[0] * self.phi_std + self.phi_mean)
        
        return {                        # Clip to observed range, 
            "pt":  np.clip(pt,  self.pt_min,  self.pt_max),
            "eta": np.clip(eta, self.eta_min, self.eta_max),
            "phi": (phi + np.pi) % (2 * np.pi) - np.pi,
        }                               # wrap phi to [-pi, pi)