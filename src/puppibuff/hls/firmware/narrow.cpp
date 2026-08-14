#include "flowhls.h"
                                        // Cast all `accum_t` back to `state_t`
void narrow(accum_t const *from, state_arr_t to)
{
    #pragma HLS inline
    for (size_t idx = 0; idx != n_channels; ++idx)
    {
        #pragma HLS unroll
        to[idx] = static_cast<state_t>(from[idx]);
    }
}

