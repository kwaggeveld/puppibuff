#include "flowhls.h"

                                        // c1 = h * 1.5, c2 = h * 0.5
static accum_t const c1 = **c1**;
static accum_t const c2 = **c2**;

void **step_top**(accum_arr_t x_in, state_arr_t v, state_arr_t v_prev, accum_arr_t x_out)
{
    #pragma HLS pipeline
    #pragma HLS array_partition variable=x_in
    #pragma HLS array_partition variable=v
    #pragma HLS array_partition variable=v_prev
    #pragma HLS array_partition variable=x_out

                                        // x = x + c1 * v - c2 * v_prev
    for (size_t idx = 0; idx != n_channels; ++idx)
    {
        #pragma HLS unroll
        x_out[idx] = x_in[idx] + c1 * v[idx] - c2 * v_prev[idx];
    }
}

