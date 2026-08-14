#include "flowhls.h"
                                        // Two-step Adams-Bashforth, unrolled
void **sample_top**(accum_arr_t x0, accum_arr_t x_out)
{
    #pragma HLS pipeline
    #pragma HLS array_partition variable=x0
    #pragma HLS array_partition variable=x_out
                                        // The solver reads x_in and writes
                                        // x_out, and needs v and v_prev, so
                                        // both alternate over two buffers
    accum_arr_t xa, xb;
    state_arr_t xs, va, vb;
    #pragma HLS array_partition variable=xa
    #pragma HLS array_partition variable=xb
    #pragma HLS array_partition variable=xs
    #pragma HLS array_partition variable=va
    #pragma HLS array_partition variable=vb

**body**

}

