from __future__ import annotations

from puppibuff.codecs import Codec
from . import constants as c

import re
from pathlib import Path
from conifer import __version__ as conifer_version
from conifer.backends import xilinxhls

from conifer.backends.xilinxhls.writer import XilinxHLSConfig, XilinxHLSModel

#-----------------------------------------------------------------------------

#--- Utility ---

def _over_all_channels(body: str) -> str:
    """A loop over the channels, unrolled. """
    return (f"for (size_t idx = 0; idx != n_channels; ++idx)\n"
             "    {\n"
            f"        #pragma HLS unroll\n"
            f"        {body}\n"
             "    }")


def _partition(*variables: str) -> str:
    """Write out partitioning pragma for `variable`."""
    return '\n'.join(f"    #pragma HLS array_partition variable={ variable }"
                     for variable in variables)


def _define_field(step: int, calls: str) -> str:
    return f"""
void { c.FIELD_NAME(step) }(state_arr_t x, state_arr_t v) {{
    #pragma HLS pipeline

{ _partition("x", "v") }

{ calls }
}}

"""

#--- Writers: Trees ---

def ap_types_h(
    n_channels: int,
    config: XilinxHLSConfig,
    accum_precision: str,
    codec: Codec,
) -> str:
    """Write the types shared by the whole design."""
    return f"""#ifndef AP_TYPES_H_
#define AP_TYPES_H_

#include "ap_fixed.h"                   // Requires `XILINX_AP_INCLUDE` set

#include <cstddef>                      // size_t

static size_t const n_channels = { n_channels };
static size_t const n_decoded  = { codec.n_decoded };

                                        // BDTs' input/output
using state_t     = { config.input_precision };
using state_arr_t = state_t[n_channels];
                                        // Solver's input/output
using accum_t     = { accum_precision };
using accum_arr_t = accum_t[n_channels];
                                        // `decode`'s output, in physical units
using decoded_t     = { codec.decoded_precision };
using decoded_arr_t = decoded_t[n_decoded];
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
    trees = '\n'.join(_tree(tree[0], idx)
                      for idx, tree in enumerate(model.trees))

                                            # conifer: (init_predict + sum) * norm,
                                            # in that order and all in score_t
    n_trees = len(model.trees)
    scores = '\n'.join(f"  scores[{ idx :2}] = tree_{ idx :02d}.decision_function(x, split_fn);"
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
{ _partition("scores") }

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

def flowhls_h(n_steps: int) -> str:
    """Declare one field top per step, plus the solver over them."""
    field_declarations = '\n'.join(f"void { c.FIELD_NAME(step) }(state_arr_t x, state_arr_t v);"
                                     for step in range(n_steps))

    return f"""#ifndef FLOWHLS_H_
#define FLOWHLS_H_

#include "ap_types.h"

{ field_declarations  }

void narrow(accum_t const *from, state_arr_t to);
void { c.STEP_TOP }(accum_arr_t x_in, state_arr_t v, state_arr_t v_prev, accum_arr_t x_out);

void { c.SAMPLE_TOP }(accum_arr_t x0, accum_arr_t x_out);
                                        // Written by the codec, see `Codec.decode_cpp`
void { Codec.s_DECODE_TOP }(accum_arr_t x, decoded_arr_t decoded);

#endif

"""


def field_sXX_cpp(step: int, bdts_in_step: list[str]) -> str:
    """Define one step's field top. Only include this step's BDT headers."""
    includes = '\n'.join(f"#include \"bdt_grid/{ bdt_name }.h\""
                         for bdt_name in bdts_in_step)

    calls = '\n'.join(f"    v[{ idx }] = { bdt_name }::decision_function(x);"
                            for idx, bdt_name in enumerate(bdts_in_step))

    return f"#include \"flowhls.h\"\n\n{ includes }\n" + _define_field(step, calls)


#--- Writers: solver ---


def ab2_step_cpp(n_steps: int) -> str:
    """Construct the solver step function, given the number of steps."""
    step_size = 1. / (n_steps - 1)
    return f"""#include "flowhls.h"

                                        // c1 = h * 1.5, c2 = h * 0.5
static accum_t const c1 = { step_size / 2 * 3 };
static accum_t const c2 = { step_size / 2 };

void { c.STEP_TOP }(accum_arr_t x_in, state_arr_t v, state_arr_t v_prev, accum_arr_t x_out)
{{
    #pragma HLS pipeline
{ _partition("x_in", "v", "v_prev", "x_out") }

                                        // x = x + c1 * v - c2 * v_prev
    { _over_all_channels("x_out[idx] = x_in[idx] + c1 * v[idx] - c2 * v_prev[idx];") }
}}

"""


def narrow_cpp() -> str:
    return f"""#include "flowhls.h"
                                        // Cast all `accum_t` back to `state_t`
void narrow(accum_t const *from, state_arr_t to)
{{
    #pragma HLS inline
    { _over_all_channels("to[idx] = static_cast<state_t>(from[idx]);") }
}}

"""


def sample_cpp(n_steps: int) -> str:
    """Unroll `solvers.ab2_solve` over the grid.

    C++ may reuse a buffer whose lifetime has ended, hardware may not, so the
    state and the velocity each alternate between two of their own. The first
    interval has no history; passing `v` as `v_prev` makes the step
    `(c1 - c2) * v = h * v`, which is the euler start `ab2_solve` takes.
    """
    x_buffers = [ "xa", "xb" ]
    v_buffers = [ "va", "vb" ]

    body = ""
    for step in range(n_steps - 1):     # ab2 never evaluates the field at t = 1
        v      = v_buffers[step % 2]
        v_prev = v if step == 0 else v_buffers[(step - 1) % 2]

        x_in   = "x0"    if step == 0           else x_buffers[(step - 1) % 2]
        x_out  = "x_out" if step == n_steps - 2 else x_buffers[step % 2]

        start = "     // No history yet: v_prev = v" if step == 0 else ""

        body += f"""
    narrow({ x_in }, xs);
    { c.FIELD_NAME(step) }(xs, { v });
    { c.STEP_TOP }({ x_in }, { v }, { v_prev }, { x_out });{ start }
"""

    return f"""#include "flowhls.h"
                                        // Two-step Adams-Bashforth, unrolled
void { c.SAMPLE_TOP }(accum_arr_t x0, accum_arr_t x_out)
{{
    #pragma HLS pipeline
{ _partition("x0", "x_out") }
                                        // The solver reads x_in and writes
                                        // x_out, and needs v and v_prev, so
                                        // both alternate over two buffers
    accum_arr_t xa, xb;
    state_arr_t xs, va, vb;
{ _partition(*x_buffers, "xs", *v_buffers) }

{ body }

}}

"""


#--- Writers: misc. ---

def bridge_cpp(n_steps: int) -> str:
    """Emit the pybind11 bridge. Unlike conifer's, which takes one sample and is
    looped over from Python, this takes a whole batch and loops in C++.
    """
    cases = '\n'.join(" " * 8 + f"case { step }: { c.FIELD_NAME(step) }(xt, vt); break;"
                      for step in range(n_steps))

    return f"""#include "firmware/ap_types.h"
#include "firmware/flowhls.h"
    
#include <vector>
#include <stdexcept>


std::vector<double> field(size_t step, std::vector<double> const &x)
{{
    size_t const n_events = x.size() / n_channels;
    std::vector<double> v(x.size());

    for (size_t event = 0; event != n_events; ++event)
    {{
        state_arr_t xt, vt;
                                        // Cast input to state_t
        for (size_t idx = 0; idx != n_channels; ++idx)
            xt[idx] = static_cast<state_t>(x[event * n_channels + idx]);

        switch (step)
        {{
{ cases }
            default: throw std::out_of_range("No such step");
        }}
                                        // Cast output to double
        for (size_t idx = 0; idx != n_channels; ++idx)
            v[event * n_channels + idx] = static_cast<double>(vt[idx]);
    }}

    return v;
}}

std::vector<double> sample(std::vector<double> const &x0)
{{
    size_t const n_events = x0.size() / n_channels;
    std::vector<double> x1(x0.size());

    for (size_t event = 0; event != n_events; ++event)
    {{
        accum_arr_t xt, yt;
                                        // Cast noise to accum_t
        for (size_t idx = 0; idx != n_channels; ++idx)
            xt[idx] = static_cast<accum_t>(x0[event * n_channels + idx]);

        { c.SAMPLE_TOP }(xt, yt);
                                        // Cast sample to double
        for (size_t idx = 0; idx != n_channels; ++idx)
            x1[event * n_channels + idx] = static_cast<double>(yt[idx]);
    }}

    return x1;
}}


std::vector<double> decode_batch(std::vector<double> const &x)
{{
    size_t const n_events = x.size() / n_channels;
    std::vector<double> out(n_events * n_decoded);

    for (size_t event = 0; event != n_events; ++event)
    {{
        accum_arr_t xt;
        decoded_arr_t decoded;
                                        // Cast sample to accum_t
        for (size_t idx = 0; idx != n_channels; ++idx)
            xt[idx] = static_cast<accum_t>(x[event * n_channels + idx]);

        { Codec.s_DECODE_TOP }(xt, decoded);
                                        // Cast physical output to double
        for (size_t idx = 0; idx != n_decoded; ++idx)
            out[event * n_decoded + idx] = static_cast<double>(decoded[idx]);
    }}

    return out;
}}

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

PYBIND11_MODULE(conifer_bridge_{ c.BRIDGE_MODULE }, m)
{{
    m.def("field",  &field);
    m.def("sample", &sample);
    m.def("decode", &decode_batch);
}}

"""


def hls_parameters_tcl(block: str, part: str, clock_period: int | float) -> str:
    """The block's build settings, sourced by `build_hls.tcl`."""
    return f"""set top { block }
set prj_name { block }
set part { part }
set clock_period { clock_period }
set flow_target vivado
set export_format ip_catalog
set m_axi_addr64 false
set version { conifer_version }
"""


def build_hls_tcl() -> str:
    """conifer's `hls-template/build_hls.tcl`, minus the testbench (we write
    none) and reading its one source from the shared `firmware/` — two levels up,
    since every block sits in `BLOCKS_DIR` and vitis_hls runs from its directory.
    """
    return """ # Adapted from:
#################
#    HLS4ML
#################
set tcldir [file dirname [info script]]
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
add_files ../../firmware/${prj_name}.cpp -cflags "-std=c++0x"

if {$opt(reset)} {
    open_solution -reset "solution1" -flow_target ${flow_target}
} else {
    open_solution "solution1" -flow_target ${flow_target}
}

set_part ${part}
create_clock -period ${clock_period} -name default

config_interface -m_axi_addr64=${m_axi_addr64}

if {$opt(synth)} {
    csynth_design
}

if {$opt(cosim)} {
    cosim_design -trace_level all
}

if {$opt(export)} {
    export_design -vendor cern.ch -library conifer -ipname ${top} -version ${version} -format ${export_format}
}
exit
"""


def vivado_synth_tcl(block: str, part: str) -> str:
    return f"""add_files { block }/solution1/syn/vhdl
synth_design -top { block } -part { part }
report_utilization -file vivado_synth.rpt
"""


#--- Writers: EMP payload ---

HRULE = "  " + "-" * 74

                                        # emp-fwk's fixed payload interface
PAYLOAD_PREAMBLE = """library ieee;
use ieee.std_logic_1164.ALL;
use ieee.numeric_std.all;

use work.ipbus.all;
use work.emp_data_types.all;
use work.emp_project_decl.all;

use work.emp_device_decl;
use work.emp_ttc_decl.all;
use work.emp_slink_types.all;

use work.tcds2_streams_pkg;

entity emp_payload is
  port(
    clk         : in  std_logic;        -- ipbus signals
    rst         : in  std_logic;
    ipb_in      : in  ipb_wbus;
    ipb_out     : out ipb_rbus;
    clk40       : in  std_logic;
    clk_payload : in  std_logic_vector(2 downto 0);
    rst_payload : in  std_logic_vector(2 downto 0);
    clk_p       : in  std_logic;        -- data clock
    rst_loc     : in  std_logic_vector(emp_device_decl.N_REGION - 1 downto 0);
    clken_loc   : in  std_logic_vector(emp_device_decl.N_REGION - 1 downto 0);
    ctrs        : in  ttc_stuff_array;
    ttc2        : in  tcds2_streams_pkg.tcds2_ttc2;
    d           : in  ldata(4 * emp_device_decl.N_REGION - 1 downto 0);  -- data in
    q           : out ldata(4 * emp_device_decl.N_REGION - 1 downto 0);  -- data out
    gpio        : out std_logic_vector(29 downto 0);  -- IO to mezzanine connector
    gpio_en     : out std_logic_vector(29 downto 0);  -- IO to mezzanine connector (three-state enables)
    slink_q     : out slink_input_data_quad_array(SLINK_MAX_QUADS-1 downto 0);
    tts_busy    : out std_logic;
    backpressure : in std_logic_vector(SLINK_MAX_QUADS-1 downto 0);
    -- External electrical signals (e.g. Apollo HDMI, Serenity HDR, RTM Zone 3)
    ext_lvds_in  : in  std_logic_vector(emp_device_decl.N_LVDS_IN - 1 downto 0);
    ext_lvds_out : out std_logic_vector(emp_device_decl.N_LVDS_OUT - 1 downto 0)
    );
end emp_payload;
"""


def _ap_fixed(precision: str) -> tuple[int, int]:
    """Extract (word, integer) bit widths of an `ap_fixed<word, integer>`."""
    match = re.fullmatch(r"ap_fixed<\s*(\d+)\s*,\s*(\d+)\s*>", precision)

    if match is None:
        raise ValueError(f"Cannot read bit widths off { precision }.")

    return int(match[1]), int(match[2])


def _port_map(n_ports: int, **ports: str) -> str:
    """Associate HLS's scalarised array ports, one element at a time."""
    return ',\n'.join(f"    { port }_{ idx } => { signal }({ idx })"
                      for port, signal in ports.items()
                      for idx in range(n_ports))


def _instance(entity: str, ports: str, suffix: str = "") -> str:
    """One block, free-running on the payload clock."""
    return f"""  inst_{ entity }{ suffix } : entity work.{ entity }
  port map(
    ap_clk => clk_p,                    -- Use the payload clock
    ap_rst => '0',                      -- Never reset
    ap_start => '1',                    -- Always start
{ ports }
  );
"""


def emp_payload_vhd(
    n_steps: int,
    n_channels: int,
    latencies: dict[str, int],
    config: XilinxHLSConfig,
    accum_precision: str,
    codec: Codec,
) -> str:
    """Tie the synthesised blocks together into a VHDL script for the emp-fwk,
    like `sample_cpp` does into C++ for simulation. 

    Every block is free-running and fully pipelined, so the design is one long
    pipeline. Each signal needs to be held until the block reading it
    catches up; these latencies are from  `utils.block_latency`.
    """
    state_width, state_int = _ap_fixed(config.input_precision)                # type: ignore
    accum_width, accum_int = _ap_fixed(accum_precision)
    decoded_width, _       = _ap_fixed(codec.decoded_precision)

    if state_int != accum_int:          # `narrow` is then a plain slice of
        raise ValueError(               # the accumulator's leading bits
            f"`state_t` and `accum_t` must share an integer width. Received widths "
            f"{ config.input_precision } and { accum_precision }."
        )
                                        # ab2 never evaluates the field at t = 1
    steps         = range(n_steps - 1)
    field_latency = [ latencies[c.FIELD_NAME(step)] for step in steps ]
    ab2_latency   = latencies[c.STEP_TOP]

                                        # A velocity is read again by the next
                                        # step's solver, as its v_prev. The
                                        # last one is never read
    hold = [ ab2_latency + field_latency[step + 1] if step + 1 in steps else 0
             for step in steps ]

                                        # The first interval has no history, so
                                        # v_prev = v: Euler step
    v_prev = [ f"s{ step :02d}_q(0)" if step == 0 else
               f"s{ step - 1 :02d}_q(HOLD({ step - 1 }))" for step in steps ]

    x_out  = [ "x_out" if step == steps[-1] else f"x{ step + 1 :02d}(0)"
               for step in steps ]

    instance_list = [
        f"""{ HRULE }
  -- Step { step :02d}

  -- Field block
{ 
    _instance(c.FIELD_NAME(step), _port_map(n_channels, x = f"s{ step :02d}_d", 
                                                        v = f"s{ step :02d}_q(0)")) 
}

  -- Sampler step block
{ 
    _instance(c.STEP_TOP, _port_map(n_channels, 
                                    x_in   = f"x{ step :02d}(FIELD_LATENCY({ step }))",
                                    v      = f"s{ step :02d}_q(0)",
                                    v_prev = v_prev[step],
                                    x_out  = x_out[step]), 
              f"_{ step :02d}") 
}
"""      
        for step in steps
    ]   # instance_list
    instances = ''.join(instance_list)

    channels = f"0 to { n_channels - 1 }"
    total    = sum(field_latency) + len(steps) * ab2_latency + latencies[Codec.s_DECODE_TOP]

    field_table = ', '.join(f"{ step } => { field_latency[step] }" for step in steps)
    hold_table  = ', '.join(f"{ step } => " + (f"AB2_LATENCY + FIELD_LATENCY({ step + 1 })"
                                               if hold[step] else "0")
                            for step in steps)

    declarations = '\n'.join(
        f"  signal x{ step :02d}   : accum_arr2d(0 to FIELD_LATENCY({ step }))({ channels });\n"
        f"  signal s{ step :02d}_d : state_arr1d({ channels });\n"
        f"  signal s{ step :02d}_q : state_arr2d(0 to HOLD({ step }))({ channels });\n"
        for step in steps)

    inputs = '\n'.join(
        f"  x00(0)({ idx }) <= d({ c.IN_BASE + idx }).data({ accum_width - 1 } downto 0);"
        for idx in range(n_channels))

    outputs = '\n'.join(
        f"  q({ c.OUT_BASE + idx }).data({ decoded_width - 1 } downto 0) <= decoded({ idx });"
        for idx in range(codec.n_decoded))
                  
    pipes = '\n'.join(
        [ f"      x{ step :02d}(1 to FIELD_LATENCY({ step })) <= "
          f"x{ step :02d}(0 to FIELD_LATENCY({ step }) - 1);"
          for step in steps if field_latency[step] ]    # A zero-cycle block needs no pipeline
      + [ f"      s{ step :02d}_q(1 to HOLD({ step })) <= "
          f"s{ step :02d}_q(0 to HOLD({ step }) - 1);"
          for step in steps if hold[step] ])

    narrow = '\n'.join(
        f"    s{ step :02d}_d(idx) <= x{ step :02d}(0)(idx)"
        f"({ accum_width - 1 } downto { accum_width - state_width });"
        for step in steps)

    decode = _instance(Codec.s_DECODE_TOP,
                       _port_map(n_channels, x = "x_out") + ",\n"
                       + _port_map(codec.n_decoded, decoded = "decoded"))

    return f"""-- Written by `puppibuff`. 
-- End to end: { total } clock cycles over { n_steps - 1 } integration steps.

{ PAYLOAD_PREAMBLE }
architecture rtl of emp_payload is

  -- Extracted from each block's csynth report
  constant AB2_LATENCY   : integer := { ab2_latency };
  constant FIELD_LATENCY : integer_vector(0 to { n_steps - 2 }) := ({ field_table });

  -- How long a velocity is held to become the next step's v_prev
  constant HOLD : integer_vector(0 to { n_steps - 2 }) := ({ hold_table });

  type state_arr1d   is array (natural range <>) of std_logic_vector({ state_width - 1 } downto 0);
  type accum_arr1d   is array (natural range <>) of std_logic_vector({ accum_width - 1 } downto 0);
  type decoded_arr1d is array (natural range <>) of std_logic_vector({ decoded_width - 1 } downto 0);

  type state_arr2d is array (natural range <>) of state_arr1d;
  type accum_arr2d is array (natural range <>) of accum_arr1d;

  -- x<s> enters step <s>, s<s>_d is its narrowed copy, 
  -- s<s>_q the velocity at that step predicted by the BDTs.
{ declarations }
  signal x_out   : accum_arr1d({ channels });
  signal decoded : decoded_arr1d(0 to { codec.n_decoded - 1 });

begin

  -- emp-fwk generic things, tying off various unused ports
  ipb_out <= IPB_RBUS_NULL;
  gpio    <= (others => '0');
  gpio_en <= (others => '0');

  -- Noise arrives on { n_channels } links from { c.IN_BASE }
{ inputs }

  -- The { codec.n_decoded } values of one sample leave on links from { c.OUT_BASE }
{ outputs }

  -- Hold every signal until the block that reads it catches up
  pipe_proc : process(clk_p)
  begin
    if rising_edge(clk_p) then
{ pipes }
    end if;
  end process;

  -- Narrow accum_t -> state_t, dropping the low { accum_width - state_width } fractional bits
  GenNarrow:
  for idx in { channels } generate
{ narrow }
  end generate;

{ instances }{ HRULE }
  -- Decoding

{ decode }
end rtl;
"""


def build_all_sh(blocks: list[str]) -> str:
    """Script to run every block's own `build_hls.tcl`, `JOBS` at a time."""
    return ("""#!/usr/bin/env bash
# Run each block's own build_hls.tcl in its own directory, one process each.
# Each run leaves its vitis_hls.log in the block directory.
#
#     ./build_all.sh                      # all blocks, csynth only
#     ./build_all.sh ab2_step             # one block
#     ./build_all.sh export=1             # pass options through to build_hls.tcl
#
# Blocks run concurrently, JOBS at a time (JOBS=1 for sequential).
# Console output goes to <block>/build.out.
# vitis_hls still writes its own <block>/vitis_hls.log.

set -uo pipefail
cd "$(dirname "$0")/blocks"

HLS=${HLS:-vitis_hls}
JOBS=${JOBS:-4}
"""
f"ALL=({ ' '.join(blocks) })\n"
"""
# name=value goes to build_hls.tcl, anything else names a block.
blocks=()
opts=()
for arg in "$@"; do
    case "$arg" in
        *=*) opts+=("$arg") ;;
        *)   blocks+=("$arg") ;;
    esac
done
if [ ${#blocks[@]} -eq 0 ]; then
    blocks=("${ALL[@]}")
fi

pids=()
names=()
for b in "${blocks[@]}"; do
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do
        sleep 1
    done
    ( cd "$b" && "$HLS" -f build_hls.tcl ${opts[@]+"${opts[@]}"} ) >"$b/build.out" 2>&1 &
    pids+=($!)
    names+=("$b")
    echo "Launched $b (pid $!)"
done

failed=()
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "--- ${names[$i]} done"
    else
        echo "--- ${names[$i]} FAILED (see blocks/${names[$i]}/build.out)"
        failed+=("${names[$i]}")
    fi
done

if [ ${#failed[@]} -ne 0 ]; then
    echo "Failed: ${failed[*]}"
    exit 1
fi
echo "built: ${blocks[*]}"
""")