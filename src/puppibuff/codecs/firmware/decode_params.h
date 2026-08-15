#ifndef DECODE_PARAMS_H_
#define DECODE_PARAMS_H_

#include "ap_types.h"
                                        // Slots per event
static size_t const multiplicity = **multiplicity**;

                                        // Fitted statistics.
static accum_t   const pt_mean  = **pt_mean**;
static accum_t   const pt_std   = **pt_std**;
static decoded_t const eta_mean = **eta_mean**;
static decoded_t const eta_std  = **eta_std**;
                                        // phi's, in turns rather than radians
static accum_t   const phi_mean = **phi_mean**;
static accum_t   const phi_std  = **phi_std**;

                                        // Observed ranges 
static decoded_t const eta_min = **eta_min**;
static decoded_t const eta_max = **eta_max**;

static decoded_t const two_pi = **two_pi**;

// Lookup table design adapted from:
// https://github.com/fastmachinelearning/hls4ml/blob/main/hls4ml/templates/vivado/nnet_utils/nnet_activation.h
static size_t const expm1_scaling    = **expm1_scaling**;
static size_t const expm1_table_size = **expm1_table_size**;

                                        // `phi` is in units of turns, not
                                        // radians. With zero integer bits the
                                        // range is [-0.5, 0.5], so overflow is
                                        // the wrap. +4 > log2(2pi)
using turns_t = ap_fixed<**turns_width**, 0>;

                                        // Entry idx is expm1(idx / expm1_scaling),
                                        // floored at pt_min.
static decoded_t const expm1_table[expm1_table_size] = {
**expm1_table**
};

#endif

