from __future__ import annotations

from puppibuff.solvers import ab2_solve

from pathlib import Path
from conifer.backends import xilinxhls

from conifer.backends.xilinxhls.writer import XilinxHLSConfig, XilinxHLSModel

#-----------------------------------------------------------------------------

SAMPLE_SOLVER = ab2_solve

#--- Naming ---

def field_name(step: int) -> str:
    """Construct function name for each field step."""
    return f"flowhls_field_s{step:02d}"


def _solver_name() -> str:
    return "ab2"


def _step_name() -> str:
    """Return solver step name."""
    return f"{_solver_name()}_step"


def _sample_name() -> str:
    """Return function name for sampling."""
    return f"{_solver_name()}_sample"

#--- Utility ---

def _over_all_channels(body: str) -> str:
    """A loop over the channels, unrolled. """
    return (f"for (size_t idx = 0; idx != n_channels; ++idx) {{\n"
            f"        #pragma HLS unroll\n"
            f"        {body}\n"
             "    }")


def _define_field(step: int, calls: str) -> str:
    return f"""
void { field_name(step) }(state_arr_t x, state_arr_t v) {{
    #pragma HLS pipeline
    #pragma HLS array_partition variable=x
    #pragma HLS array_partition variable=v
{ calls }
}}

"""

#--- Writers: Trees ---

def ap_types_h(n_channels: int, config: XilinxHLSConfig, accum_precision: str) -> str:
    """Write the types shared by the whole design."""
    return f"""#ifndef AP_TYPES_H_
#define AP_TYPES_H_

#include "ap_fixed.h"                   // Requires `XILINX_AP_INCLUDE` set

static size_t const n_channels = { n_channels };

                                        // BDTs' input/output
using state_t     = { config.input_precision };
using state_arr_t = state_t[n_channels];
                                        // Solver's input/output
using accum_t     = { accum_precision };
using accum_arr_t = accum_t[n_channels];
                                        // Conifer's precisions
using threshold_t = { config.threshold_precision };
using weight_t    = { config.weight_precision };
using score_t     = { config.score_precision } ;

#endif

"""


TREE_FIELDS = ["feature", "threshold", "value",
               "children_left", "children_right", "parent"]

def _tree(tree, idx: int) -> str:
    """One `BDT::Tree` instance, formatted as conifer's writer formats it, minus
    the `weight` table `bdt_h` deletes.
    """
    rows = [ f"    {{ {', '.join(map(str, getattr(tree, field)))} }}"
             for field in TREE_FIELDS ]

                                            # A BDT reads the whole state vector,
                                            # so n_features == n_channels.
                                            # We emit the symbol rather than a second
                                            # copy of the number
    return (f"static BDT::Tree<{ idx }, { tree.n_nodes() }, { tree.n_leaves() }, "
            f"n_channels, state_arr_t, score_t, weight_t, threshold_t> "
            f"const tree_{ idx :02d} =\n{{\n" + ",\n".join(rows) + "\n};")


def bdt_sXX_gXX_h(model: XilinxHLSModel) -> str:
    trees = "\n".join(_tree(tree[0], idx)
                      for idx, tree in enumerate(model.trees))

                                            # conifer: (init_predict + sum) * norm,
                                            # in that order and all in score_t
    n_trees = len(model.trees)
    scores = "\n".join(f"  scores[{ idx :2}] = tree_{ idx :02d}.decision_function(x, split_fn);"
                       for idx in range(n_trees))

    name = model.config.project_name                                          # type: ignore[reportAttributeAccessIssue]
    init_predict = model.init_predict[0]                                      # type: ignore[reportAttributeAccessIssue]
    norm = model.norm                                                         # type: ignore[reportAttributeAccessIssue]

    return f"""#ifndef {name.upper()}_H_
#define { name.upper() }_H_

#include "BDT.h"
#include "../ap_types.h"

namespace { name }
{{

static size_t const n_trees = { n_trees };

static score_t const init_predict  = { init_predict };
static score_t const normalisation = { norm };

                                        // Only deduces BDT::Tree's T/U template
                                        // type parameter. The comparison
                                        // itself is hardcoded into BDT.h,
                                        // see `puppibuff/hls/bdt_h.py`
inline bool split_fn(state_t const *const a, threshold_t const *const b)
{{
    return *a < *b;
}}

{ trees }

inline score_t decision_function(state_arr_t x)
{{
    #pragma HLS pipeline
    score_t scores[n_trees];
    #pragma HLS array_partition variable=scores

{ scores }

    BDT::OpAdd<score_t> op_add;
    score_t score = init_predict;
    score += BDT::reduce<score_t, n_trees, BDT::OpAdd<score_t>>(scores, op_add);
    score *= normalisation;

    return score;
}}

}}   // { name }
#endif

"""

#--- Writers: BDTs ---

SPLIT_CALL  = "comparison[i] = split_fn(&accumulation, &threshold[i]);"
SPLIT_PATCH = "comparison[i] = accumulation < threshold[i];"

OBLIQUE_DOT = """        for(int i_feat = 0; i_feat < n_features; i_feat++ ){
          accumulation += x[i_feat] * weight[i][i_feat];
        }"""
AXIS_SELECT = "        accumulation = x[feature[i]];"

WEIGHT_TABLE = "  weight_t weight[n_nodes][n_features];\n"

def bdt_h(unroll: bool) -> str:
    """Retrieve conifer's default `BDT.h` and patch it in three places.

    1. Replace the call through `split_fn` with the comparison itself.
       conifer passes the split convention as a function pointer, and the
       `#pragma HLS pipeline` on `Tree::decision_function` keeps that function
       out of line, so csynth sees an indirect call and rejects it. The
       convention is fixed by conifer's converter rather than by the model.
    2. Replace the x.weight dot product with an indexing operation.
       conifer supports oblique trees, but puppibuff does not use them.
       Weights are thus Kronecker deltas, collapsing the x.weights dot product
       (written by conifer) to an entry selection. Writing this directly saves
       many resources.
    3. Delete the now-unused one-hot weight table from the `Tree` struct. Patch
       #2 made it obsolete. Note the tree tables are aggregate initialisers, so
       the corresponding writer must drop its weight row as well.

    Diff against conifer's BDT_unrolled.h:
        <   weight_t weight[n_nodes][n_features];
        ---
        > (deleted)

        <         for(int i_feat = 0; i_feat < n_features; i_feat++ ){
        <           accumulation += x[i_feat] * weight[i][i_feat];
        <         }
        <         comparison[i] = split_fn(&accumulation, &threshold[i]);
        ---
        >         accumulation = x[feature[i]];
        >         comparison[i] = accumulation < threshold[i];

    """
    if not unroll:                  # The rolled variant reads its trees from
        raise NotImplementedError(  # an array `parameters.h` we do not write
            "Merging is implemented for Unroll = True only."
        )

    bdt_h = (Path(xilinxhls.__file__).parent
             / "firmware" / "BDT_unrolled.h").read_text()

    for fragment in (SPLIT_CALL, OBLIQUE_DOT, WEIGHT_TABLE):
        if fragment not in bdt_h:
            raise RuntimeError(
                f"conifer's `BDT.h` no longer contains:\n{fragment}\n"
                "Re-check the patch fragments against this conifer version."
            )

    return (bdt_h.replace(SPLIT_CALL, SPLIT_PATCH)
                 .replace(OBLIQUE_DOT, AXIS_SELECT)
                 .replace(WEIGHT_TABLE, ""))


#--- Writers: fields ---

def flowhls_ih(names: list[list[str]]) -> str:
    """Construct the `.ih` for FlowHLS, which includes all BDT headers."""
    bdt_headers = "\n".join(f"#include \"bdt_grid/{ name }.h\""
                            for row in names for name in row)
    return f"""#include "flowhls.h"

{ bdt_headers }

"""


def flowhls_h(n_steps: int) -> str:
    """Declare one field top per step, plus the sampler over them."""
    field_declarations = "\n".join(f"void {field_name(step)}(state_arr_t x, state_arr_t v);"
                                     for step in range(n_steps))

    return f"""#ifndef FLOWHLS_H_
#define FLOWHLS_H_

#include "ap_types.h"

{ field_declarations  }

void narrow(accum_arr_t from, state_arr_t to);
void { _step_name() }(accum_arr_t x_in, state_arr_t v, state_arr_t v_prev, accum_arr_t x_out);

void { _sample_name() }(accum_arr_t x0, accum_arr_t x1);

#endif

"""


def field_sXX_cpp(step: int, bdts_in_step: list[str]) -> str:
    """Define the field top functions.Return as dict, with key the filename
    and item the source.
    """
    calls = "\n".join(f"    v[{ idx }] = { bdt_name }::decision_function(x);"
                            for idx, bdt_name in enumerate(bdts_in_step))

    return "#include \"flowhls.ih\"" + _define_field(step, calls)


def solve_step_cpp(n_steps: int) -> str:
    """Construct the solver step function, given the number of steps."""
    step_size = 1. / (n_steps - 1)
    return f"""#include "flowhls.ih"

static accum_t const c1 = { step_size / 2 * 3 };
static accum_t const c2 = { step_size / 2 };

void { _step_name() }(accum_arr_t x_in, state_arr_t v, state_arr_t v_prev, accum_arr_t x_out)
{{
    #pragma HLS array_partition variable=x_in
    #pragma HLS array_partition variable=v
    #pragma HLS array_partition variable=v_prev
    #pragma HLS array_partition variable=x_out

                                        // x = x + c1 * v - c2 * v_prev
    { _over_all_channels("x_out[idx] = x_in[idx] + c1 * v[idx] - c2 * v_prev[idx];") }
}}

"""


def narrow_cpp() -> str:
    return f"""#include "flowhls.ih"
                                        // Cast all `accum_t` back to `state_t`
void narrow(accum_arr_t from, state_arr_t to)
{{
    #pragma HLS inline
    { _over_all_channels("to[idx] = static_cast<state_t>(from[idx]);") }
}}

"""


#--- Writers: misc. ---

# TODO