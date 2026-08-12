from ..solvers import ab2_solve

#-----------------------------------------------------------------------------

#--- Target chip properties ---

XILINX_PART     = "xcvu13p-flga2577-2-e"
CLOCK_PERIOD    = 2.5                   # In ns, the payload clock (400 MHz)
ACCUM_PRECISION = "ap_fixed<32,8>"      # The solver's state, wider than score_t

IN_BASE    = 64                         # First input link
OUT_BASE   = 48                         # First output link

#--- Naming ---

SOLVER     = "ab2"
STEP_TOP   = f"{ SOLVER }_step"
SAMPLE_TOP = f"{ SOLVER }_sample"
FIELD_NAME = lambda step: f"field_s{ step :02d}"

SAMPLE_SOLVER = ab2_solve               # The scheme `write.sample_cpp` unrolls,
                                        # which `FlowHLS.sample` checks for

BRIDGE_MODULE = "flowhls"               # The merged design's pybind11 bridge

#--- Disk locations ---

BLOCKS_DIR   = "blocks"                 # One HLS project per block of the design
BDT_DATA     = "bdt_data"               # And the folder holding its BDT `.json`s

BUILD_SCRIPT = "build_all.sh"           # Synthesises every block
PAYLOAD_FILE = "emp_payload.vhd"
CODEC_FILE   = "codec.json"             # `write` saves it beside the design, so
                                        # everything after recovers it like
                                        # `output_dir`.

LATENCY_KEY = "PerformanceEstimates/SummaryOfOverallLatency/Worst-caseLatency"
