#ifndef **name_upper**_HH_
#define **name_upper**_HH_

#include "BDT.h"
#include "../ap_types.h"

namespace **name**
{

static size_t const n_trees = **n_trees**;

static score_t const init_predict  = **init_predict**;
static score_t const normalisation = **normalisation**;

                                        // Only deduces BDT::Tree's T/U template
                                        // type parameter. The comparison
                                        // itself is hardcoded into BDT.h,
                                        // see `puppibuff/hls/write.py`
inline bool split_fn(state_t const *const a, threshold_t const *const b)
{
    return *a < *b;
}

**trees**

inline score_t decision_function(state_arr_t x)
{
    #pragma HLS pipeline
    score_t scores[n_trees];
    #pragma HLS array_partition variable=scores

**scores**

    BDT::OpAdd<score_t> op_add;
    score_t score = init_predict;
    score += BDT::reduce<score_t, n_trees, BDT::OpAdd<score_t>>(scores, op_add);
    score *= normalisation;

    return score;
}

}   // namespace **name**
#endif

