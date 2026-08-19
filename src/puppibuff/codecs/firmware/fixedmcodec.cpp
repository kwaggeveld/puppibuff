#include "flowhls.hh"
#include "codec_params.hh"

/* Every fitted constant lives in `codec_params.hh` so this file is plain source.
 */

inline decoded_t expm1_lookup(accum_t value)
{
    #pragma HLS inline
    int table_index = value * expm1_scaling;

    if (table_index < 0)
        table_index = 0;
    else if (table_index > int(expm1_table_size) - 1)
        table_index = expm1_table_size - 1;

    return expm1_table[table_index];
}

inline decoded_t clip(decoded_t value, decoded_t low, decoded_t high)
{
    #pragma HLS inline
    return value < low  ? low  :
           value > high ? high : value;
}

inline decoded_t wrap_phi(accum_t value)
{
    #pragma HLS inline                  // Narrowing wraps back to [-0.5, 0.5]
    turns_t const turns = value * phi_std + phi_mean;
    return turns * two_pi;
}
                                        // Name fixed by `Codec.s_DECODE_TOP`,
                                        // which `flowhls.hh` declares
void decode(accum_arr_t x, decoded_arr_t decoded)
{
    #pragma HLS pipeline
    #pragma HLS array_partition variable=x
    #pragma HLS array_partition variable=decoded

    for (size_t idx = 0; idx != multiplicity; ++idx)
    {
        #pragma HLS unroll
        decoded_t const pt  = expm1_lookup(x[idx] * pt_std + pt_mean);
        decoded_t const eta = x[multiplicity + idx] * eta_std + eta_mean;
        decoded_t const phi = wrap_phi(x[2 * multiplicity + idx]);

        decoded[idx]                    = pt;           // The LUT clips
        decoded[multiplicity + idx]     = clip(eta, eta_min, eta_max);
        decoded[2 * multiplicity + idx] = phi;          // `wrap_phi` rescales
    }
}

