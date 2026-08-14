from __future__ import annotations

from puppibuff.codecs import Codec
from ..utils import fill_template
from . import constants as c

import re
from pathlib import Path
from conifer import __version__ as conifer_version
from conifer.backends import xilinxhls

from conifer.backends.xilinxhls.writer import XilinxHLSConfig, XilinxHLSModel

#-----------------------------------------------------------------------------

def _template(name: str, /, **fields) -> str:
    """Fill one of this package's firmware templates."""
    return fill_template("puppibuff.hls", name, **fields)


#--- Writers: Trees ---

def ap_types_h(
    n_channels: int,
    config: XilinxHLSConfig,
    accum_precision: str,
    codec: Codec,
) -> str:
    """Write the types shared by the whole design."""
    return _template("ap_types.h",
        n_channels  = n_channels,
        n_decoded   = codec.n_decoded,
        state_t     = config.input_precision,                                   # type: ignore[reportAttributeAccessIssue]
        accum_t     = accum_precision,
        decoded_t   = codec.decoded_precision,
        threshold_t = config.threshold_precision,                               # type: ignore[reportAttributeAccessIssue]
        weight_t    = config.weight_precision,                                  # type: ignore[reportAttributeAccessIssue]
        score_t     = config.score_precision,                                   # type: ignore[reportAttributeAccessIssue]
    )


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

    name = model.config.project_name                                            # type: ignore[reportAttributeAccessIssue]

    return _template("bdt.h",
        name          = name,
        name_upper    = name.upper(),
        n_trees       = n_trees,
        init_predict  = model.init_predict[0],                                  # type: ignore[reportAttributeAccessIssue]
        normalisation = model.norm,                                             # type: ignore[reportAttributeAccessIssue]
        trees         = trees,
        scores        = scores,
    )

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

    return _template("flowhls.h",
        field_declarations = field_declarations,
        step_top           = c.STEP_TOP,
        sample_top         = c.SAMPLE_TOP,
        decode_top         = Codec.s_DECODE_TOP,
    )


def field_sXX_cpp(step: int, bdts_in_step: list[str]) -> str:
    """Define one step's field top. Only include this step's BDT headers."""
    includes = '\n'.join(f"#include \"bdt_grid/{ bdt_name }.h\""
                         for bdt_name in bdts_in_step)

    calls = '\n'.join(f"    v[{ idx }] = { bdt_name }::decision_function(x);"
                            for idx, bdt_name in enumerate(bdts_in_step))

    return _template("field.cpp",
        field_name = c.FIELD_NAME(step),
        includes   = includes,
        calls      = calls
    )


#--- Writers: solver ---


def ab2_step_cpp(n_steps: int) -> str:
    """Construct the solver step function, given the number of steps."""
    step_size = 1. / (n_steps - 1)

    return _template("ab2_step.cpp",
        step_top = c.STEP_TOP,
        c1       = step_size / 2 * 3,
        c2       = step_size / 2
    )


def narrow_cpp() -> str:
    return _template("narrow.cpp")


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

    return _template("sample.cpp", sample_top = c.SAMPLE_TOP, body = body)


#--- Writers: misc. ---

def bridge_cpp(n_steps: int) -> str:
    """Emit the pybind11 bridge. Unlike conifer's, which takes one sample and is
    looped over from Python, this takes a whole batch and loops in C++.
    """
    cases = '\n'.join(" " * 12 + f"case { step }: { c.FIELD_NAME(step) }(xt, vt); break;"
                      for step in range(n_steps))

    return _template("bridge.cpp",
        cases         = cases,
        sample_top    = c.SAMPLE_TOP,
        decode_top    = Codec.s_DECODE_TOP,
        bridge_module = c.BRIDGE_MODULE
    )


def hls_parameters_tcl(block: str, part: str, clock_period: int | float) -> str:
    """The block's build settings, sourced by `build_hls.tcl`."""
    return _template("hls_parameters.tcl",
        block        = block,
        part         = part,
        clock_period = clock_period,
        version      = conifer_version
    )


def build_hls_tcl() -> str:
    """conifer's `hls-template/build_hls.tcl`, minus the testbench (we write
    none) and reading its one source from the shared `firmware/` — two levels up,
    since every block sits in `BLOCKS_DIR` and vitis_hls runs from its directory.
    """
    return _template("build_hls.tcl")


def vivado_synth_tcl(block: str, part: str) -> str:
    return _template("vivado_synth.tcl", block = block, part = part)


#--- Writers: EMP payload ---

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
    like `sample_cpp` does into C++ for emulation. 

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

    hrule = "  " + "-" * 74             # Also literal in `emp_payload.vhd`

    instance_list = [
        f"""{ hrule }
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

    channels = f"0 to { n_channels - 1 }"
    total    = sum(field_latency) + len(steps) * ab2_latency + latencies[Codec.s_DECODE_TOP]

    return _template(
        "emp_payload.vhd",

        total       = total,
        n_intervals = n_steps - 1,

        ab2_latency   = ab2_latency,
        last_interval = n_steps - 2,

        field_table = ', '.join(f"{ step } => { field_latency[step] }" for step in steps),
        hold_table  = ', '.join(f"{ step } => " + (f"AB2_LATENCY + FIELD_LATENCY({ step + 1 })"
                                                   if hold[step] else "0")
            for step in steps
        ),
        state_msb    = state_width - 1,
        accum_msb    = accum_width - 1,
        decoded_msb  = decoded_width - 1,
        dropped_bits = accum_width - state_width,

        channels     = channels,
        last_decoded = codec.n_decoded - 1,
        n_channels   = n_channels,
        n_decoded    = codec.n_decoded,
        in_base      = c.IN_BASE,
        out_base     = c.OUT_BASE,

        declarations = '\n'.join(
            f"  signal x{ step :02d}   : accum_arr2d(0 to FIELD_LATENCY({ step }))({ channels });\n"
            f"  signal s{ step :02d}_d : state_arr1d({ channels });\n"
            f"  signal s{ step :02d}_q : state_arr2d(0 to HOLD({ step }))({ channels });\n"
            for step in steps
        ),
        inputs = '\n'.join(
            f"  x00(0)({ idx }) <= d({ c.IN_BASE + idx }).data({ accum_width - 1 } downto 0);"
            for idx in range(n_channels)
        ),
        outputs = '\n'.join(
            f"  q({ c.OUT_BASE + idx }).data({ decoded_width - 1 } downto 0) <= decoded({ idx });"
            for idx in range(codec.n_decoded)
        ),
        pipes = '\n'.join(
            [ f"      x{ step :02d}(1 to FIELD_LATENCY({ step })) <= "
              f"x{ step :02d}(0 to FIELD_LATENCY({ step }) - 1);"
              for step in steps if field_latency[step] ]    # A zero-cycle block needs no pipeline
          + [ f"      s{ step :02d}_q(1 to HOLD({ step })) <= "
              f"s{ step :02d}_q(0 to HOLD({ step }) - 1);"
              for step in steps if hold[step] ]
        ),
        narrow = '\n'.join(
            f"    s{ step :02d}_d(idx) <= x{ step :02d}(0)(idx)"
            f"({ accum_width - 1 } downto { accum_width - state_width });"
            for step in steps
        ),
        instances = ''.join(instance_list),
        decode = _instance(Codec.s_DECODE_TOP,
                           _port_map(n_channels, x = "x_out") + ",\n"
                           + _port_map(codec.n_decoded, decoded = "decoded")),
    )


def build_all_sh(blocks: list[str]) -> str:
    """Script to run every block's own `build_hls.tcl`, `JOBS` at a time."""
    return _template("build_all.sh", blocks = ' '.join(blocks))
