#ifndef AP_TYPES_H_
#define AP_TYPES_H_

#include "ap_fixed.h"                   // Requires `XILINX_AP_INCLUDE` set

#include <cstddef>                      // size_t

static size_t const n_channels = **n_channels**;
static size_t const n_decoded  = **n_decoded**;

                                        // BDTs' input/output
using state_t     = **state_t**;
using state_arr_t = state_t[n_channels];
                                        // Solver's input/output
using accum_t     = **accum_t**;
using accum_arr_t = accum_t[n_channels];
                                        // `decode`'s output, in physical units
using decoded_t     = **decoded_t**;
using decoded_arr_t = decoded_t[n_decoded];
                                        // Conifer's precisions
using threshold_t = **threshold_t**;
using weight_t    = **weight_t**;
using score_t     = **score_t** ;

#endif

