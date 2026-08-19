#include "flowhls.hh"

**includes**

void **field_name**(state_arr_t x, state_arr_t v) {
    #pragma HLS pipeline

    #pragma HLS array_partition variable=x
    #pragma HLS array_partition variable=v

**calls**
}

