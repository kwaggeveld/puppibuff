from __future__ import annotations

from ..solvers import midpoint_solve

from conifer.model import ModelBase

# conifer's HLS output is written one model per translation unit: `parameters.h`
# puts every constant, typedef and tree at global scope behind a fixed include
# guard, and `BDT.cpp` supplies the ensemble body as
#   template<> void BDT::BDT<n_trees, n_classes, n_features, ...>::tree_scores(...)
# An explicit specialisation may only be declared in the template's own namespace,
# so it cannot be moved into a per-BDT namespace; and since every BDT of the grid
# shares n_trees/n_classes/n_features and its precisions, hoisting it out leaves
# two BDTs with the *identical* specialisation signature. Namespacing alone
# therefore cannot merge the grid.
# So we re-emit the thin `BDT::BDT` wrapper as a plain function inside a per-BDT
# namespace, and reuse `BDT.h` verbatim: `BDT::Tree::decision_function` (the
# kernel, with its pragmas), `BDT::reduce` and `BDT::OpAdd` are untouched.

#-----------------------------------------------------------------------------

# Current state: mostly written by Claude. Will polish later.

TREE_FIELDS = ["feature", "weight", "threshold", "value",
               "children_left", "children_right", "parent"]

                                        # XGBoost's split convention, which conifer
                                        # writes as a module-level constant
                                        # (`converters.splitting_conventions`), so it
                                        # is a fact about the library, not the model
SPLIT = "<"

                                        # The design's entry points. An HLS project
                                        # synthesises one top at a time, so
                                        # `FlowHLS.flowhls_top` picks between them
SAMPLE_TOP = "flowhls_sample"

                                        # The scheme `_sample_top` unrolls, and so
                                        # the only one a compiled merged design can
                                        # sample with. `FlowHLS.sample` guards on it
SAMPLE_SOLVER = midpoint_solve


def field_top(step: int) -> str:
    return f"flowhls_field_s{step:02d}"


def _tree(tree, index: int) -> str:
    """One `BDT::Tree` instance, formatted as conifer's writer formats it."""
    rows = []
    for field in TREE_FIELDS:
        row = ",".join(map(str, getattr(tree, field)))
        if field == "weight":               # List of lists -> braced initialiser
            row = row.replace("[", "{").replace("]", "}")

        rows.append(f"    {{{row}}}")

                                            # A BDT reads the whole state vector,
                                            # so its n_features *is* n_channels —
                                            # emit the symbol rather than a second
                                            # copy of the number
    return (f"static const BDT::Tree<{index}, {tree.n_nodes()}, {tree.n_leaves()}, "
            f"n_channels, state_arr_t, score_t, weight_t, threshold_t> "
            f"tree_{index} = {{\n" + ",\n".join(rows) + "\n};")


def tree_header(model: ModelBase, name: str) -> str:
    """Emit one BDT as a self-contained namespace in a header.

    Everything is `static const`/`inline`, so the header can be included by both
    the merged top and the bridge without duplicate symbols. Assumes a
    single-output regressor, which `convert_grid`'s `multi_output` guard ensures.
    """
    trees = "\n".join(_tree(tree[0], index)
                      for index, tree in enumerate(model.trees))

                                            # conifer: (init_predict + sum) * norm,
                                            # in that order and all in score_t
    scores = "\n".join(f"  scores[{index}] = tree_{index}.decision_function(x, split_fn);"
                       for index in range(model.n_trees))

    return f"""#ifndef {name.upper()}_H_
#define {name.upper()}_H_

#include "BDT.h"
#include "ap_types.h"

namespace {name} {{

static const int n_trees = {model.n_trees};

static const score_t init_predict  = {model.init_predict[0]};
static const score_t normalisation = {model.norm};

                                        // Only deduces BDT::Tree's T/U — the
                                        // comparison itself is baked into BDT.h,
                                        // since HLS cannot synthesise a call
                                        // through a pointer (`bdt_h_patch`)
inline bool split_fn(const state_t* a, const threshold_t* b) {{
  return *a {SPLIT} *b;
}}

{trees}

inline score_t decision_function(state_arr_t x) {{
  #pragma HLS pipeline
  score_t scores[n_trees];
  #pragma HLS array_partition variable=scores
{scores}
  BDT::OpAdd<score_t> op_add;
  score_t score = init_predict;
  score += BDT::reduce<score_t, n_trees, BDT::OpAdd<score_t>>(scores, op_add);
  score *= normalisation;

  return score;
}}

}}
#endif
"""


                                        # The two pieces of conifer's kernel we
                                        # cannot take verbatim, and their fixed forms
SPLIT_CALL   = "comparison[i] = split_fn(&accumulation, &threshold[i]);"
SPLIT_DIRECT = f"comparison[i] = accumulation {SPLIT} threshold[i];"

OBLIQUE_DOT = """        for(int i_feat = 0; i_feat < n_features; i_feat++ ){
          accumulation += x[i_feat] * weight[i][i_feat];
        }"""
AXIS_SELECT = "        accumulation = x[feature[i]];"


def bdt_h_patch(source: str) -> str:
    """conifer's `BDT.h`, patched in two places. Rest remains untouched.

    Diff for conifer's `BDT.h`:

        <         for(int i_feat = 0; i_feat < n_features; i_feat++ ){
        <           accumulation += x[i_feat] * weight[i][i_feat];
        <         }
        <         comparison[i] = split_fn(&accumulation, &threshold[i]);
        ---
        >         accumulation = x[feature[i]];
        >         comparison[i] = accumulation < threshold[i];

    Both substitutions are unconditional, which assumes XGBoost stays the only
    tree provider: conifer's converter fixes both properties for the whole
    library rather than per model, so there is nothing per-model to check. A new
    provider needs them gated on it — see `SPLIT` and the one-hot note below.

    1. `Tree::decision_function` takes the split comparison as a function
       *pointer* and carries `#pragma HLS pipeline II = 1`, so HLS keeps it out
       of line and then refuses the call: "Indirect function call is not
       supported" (HLS 214-138), failing csynth for every tree. The convention is
       a compile-time fact, so it is substituted in rather than dispatched to.

    2. conifer supports oblique trees, so every node computes a full `x . weight`
       dot product. XGBoost's are axis-aligned — its converter writes a one-hot
       weight row per node, unconditionally — so that is one DSP per (internal
       node x feature) spent multiplying by 1 and 0. Measured: 3240 of 3258 DSPs,
       106% of one SLR, on a 4x3 grid of 20-tree depth-2 BDTs. Selecting the
       feature directly drops all of them, and drops the weight table with them.
       Applying it to genuinely oblique trees would emit silently wrong firmware,
       so a new provider needs a per-node one-hot check gating it.
    """
    for fragment in (SPLIT_CALL, OBLIQUE_DOT):
        if fragment not in source:      # conifer moved it: fail loudly rather
            raise RuntimeError(         # than silently emitting the unpatched form
                f"conifer's BDT.h no longer contains:\n{fragment}\n"
                "re-check the patch fragments against this conifer version"
            )

    source = source.replace(SPLIT_CALL, SPLIT_DIRECT)
    source = source.replace(OBLIQUE_DOT, AXIS_SELECT)

    return source


def ap_types_h(n_channels: int, config, accum_precision: str) -> str:
    """Write the types shared by the whole design."""
    return f"""#ifndef AP_TYPES_H_
#define AP_TYPES_H_

#include "ap_fixed.h"

static const int n_channels = {n_channels};

                                        // What the BDTs read: conifer's input_t
typedef {config.input_precision} state_t;
typedef state_t state_arr_t[n_channels];

typedef {config.threshold_precision} threshold_t;
typedef {config.weight_precision} weight_t;
typedef {config.score_precision} score_t;

                                        // Wider than score_t: the solver
                                        // accumulates over 2 * (n_steps - 1) adds,
                                        // and carries the sampler's own I/O so
                                        // x0 is not quantised before integrating
typedef {accum_precision} accum_t;
typedef accum_t accum_arr_t[n_channels];

#endif
"""


def flowhls_h(n_steps: int) -> str:
    """Declare one field top per step, plus the sampler over them."""
    tops = "\n".join(f"void {field_top(step)}(state_arr_t x, state_arr_t v);"
                     for step in range(n_steps))

    return f"""#ifndef FLOWHLS_H_
#define FLOWHLS_H_

#include "ap_types.h"

{tops}

void {SAMPLE_TOP}(accum_arr_t x0, accum_arr_t x_out);

#endif
"""


def _channel_loop(body: str) -> str:
    """A loop over the channels, unrolled."""
    return (f"  for (int i = 0; i < n_channels; i++) {{\n"
            f"    #pragma HLS unroll\n"
            f"    {body}\n"
             "  }")


def _sample_top(n_steps: int) -> str:
    """Write `midpoint_solve` unrolled over the grid.

    Each iteration evaluates the field at `t` and `t + h/2`, which `t_to_step`
    snaps to rows k and k+1. So consecutive rows run in series and there is no
    loop to roll over. Both the step size and the state live in `accum_t`: a
    plain C++ literal would promote the update to floating point.
    """
    step_size = 1. / (n_steps - 1)

    body = "".join(f"""
  narrow(x, xs);
  {field_top(step)}(xs, v);
{_channel_loop("x_mid[i] = x[i] + half_step * v[i];")}

  narrow(x_mid, xs);
  {field_top(step + 1)}(xs, v);
{_channel_loop("x[i] = x[i] + step_size * v[i];")}""" for step in range(n_steps - 1))

    return f"""static const accum_t step_size = {step_size!r};
static const accum_t half_step = {step_size / 2!r};

static void narrow(const accum_t from[n_channels], state_arr_t to) {{
  #pragma HLS inline
{_channel_loop("to[i] = (state_t) from[i];")}
}}

void {SAMPLE_TOP}(accum_arr_t x0, accum_arr_t x_out) {{
  #pragma HLS pipeline
  #pragma HLS array_partition variable=x0
  #pragma HLS array_partition variable=x_out
  accum_t x[n_channels], x_mid[n_channels];
  state_arr_t xs, v;
  #pragma HLS array_partition variable=x
  #pragma HLS array_partition variable=x_mid

{_channel_loop("x[i] = x0[i];")}
{body}

{_channel_loop("x_out[i] = x[i];")}
}}"""


def flowhls_cpp(names: list[list[str]]) -> str:
    """Define the field top functions. Each runs its step's BDTs on the same 
    state vector, so a step's groups are independent and synthesise in parallel.
    """
    includes = "\n".join(f'#include "{name}.h"'
                         for row in names for name in row)

    tops = []
    for step, row in enumerate(names):
        calls = "\n".join(f"  v[{group}] = {name}::decision_function(x);"
                          for group, name in enumerate(row))

        tops.append(f"""void {field_top(step)}(state_arr_t x, state_arr_t v) {{
  #pragma HLS pipeline
  #pragma HLS array_partition variable=x
  #pragma HLS array_partition variable=v
{calls}
}}""")

    tops = "\n".join(tops)                  # type: ignore

    return f"""#include "flowhls.h"
#include "ap_types.h"
{includes}

{tops}

{_sample_top(len(names))}
"""


def bridge_cpp(project: str, n_steps: int) -> str:
    """Emit the pybind11 bridge. Unlike conifer's, which takes one sample and is
    looped from Python, this takes a whole batch and loops in C++.
    """
    cases = "\n".join(f"      case {step}: {field_top(step)}(xt, vt); break;"
                      for step in range(n_steps))

    return f"""#include <vector>
#include <stdexcept>
#include "firmware/ap_types.h"
#include "firmware/flowhls.h"

std::vector<double> field(int step, const std::vector<double>& x) {{
  int n_samples = x.size() / n_channels;
  std::vector<double> y(n_samples * n_channels);

  for (int n = 0; n < n_samples; n++) {{
    state_arr_t xt, vt;
    for (int i = 0; i < n_channels; i++) {{
      xt[i] = (state_t) x[n * n_channels + i];
    }}

    switch (step) {{
{cases}
      default: throw std::out_of_range("no such step");
    }}

    for (int i = 0; i < n_channels; i++) {{
      y[n * n_channels + i] = (double) vt[i];
    }}
  }}

  return y;
}}

std::vector<double> sample(const std::vector<double>& x0) {{
  int n_samples = x0.size() / n_channels;
  std::vector<double> y(n_samples * n_channels);

  for (int n = 0; n < n_samples; n++) {{
    accum_arr_t xt, yt;
    for (int i = 0; i < n_channels; i++) {{
      xt[i] = (accum_t) x0[n * n_channels + i];
    }}

    {SAMPLE_TOP}(xt, yt);

    for (int i = 0; i < n_channels; i++) {{
      y[n * n_channels + i] = (double) yt[i];
    }}
  }}

  return y;
}}

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
PYBIND11_MODULE(conifer_bridge_{project}, m){{
  m.def("field",  &field);
  m.def("sample", &sample);
}}
"""


def hls_parameters_tcl(project: str, top: str, part: str,
                       clock_period: int | float) -> str:
    return f"""set top {top}
set prj_name {project}
set part {part}
set clock_period {clock_period}
set flow_target vivado
set export_format ip_catalog
set m_axi_addr64 false
"""


def build_hls_tcl() -> str:
    """conifer's `hls-template/build_hls.tcl`, minus the testbench
    and minus `firmware/BDT.cpp` (we don't write these).
    """
    return """set tcldir [file dirname [info script]]
source [file join $tcldir hls_parameters.tcl]

array set opt {
    reset      0
    synth      1
    cosim      0
    export     0
}

foreach arg $::argv {
  foreach o [lsort [array names opt]] {
    regexp "$o=+(\\\\w+)" $arg unused opt($o)
  }
}

if {$opt(reset)} {
    open_project -reset ${prj_name}
} else {
    open_project ${prj_name}
}

set_top ${top}
add_files firmware/flowhls.cpp -cflags "-std=c++0x"

open_solution -reset "solution1" -flow_target ${flow_target}
set_part ${part}
create_clock -period ${clock_period} -name default

if {$opt(synth)} {
    csynth_design
}

if {$opt(cosim)} {
    cosim_design -trace_level all
}

if {$opt(export)} {
    export_design -vendor cern.ch -library conifer -ipname ${top} -format ${export_format}
}
exit
"""
