#ifndef FLOWHLS_HH_
#define FLOWHLS_HH_

#include "ap_types.h"

**field_declarations**

void narrow(accum_t const *from, state_arr_t to);
void **step_top**(accum_arr_t x_in, state_arr_t v, state_arr_t v_prev, accum_arr_t x_out);

void **sample_top**(accum_arr_t x0, accum_arr_t x_out);
                                        // Written by the codec, see `Codec.decode_cpp`
void **decode_top**(accum_arr_t x, decoded_arr_t decoded);

#endif

