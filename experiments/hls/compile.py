from puppibuff.configs import FlatPuppiJetConfig
from puppibuff.hls import FlowHLS

from puppibuff.utils import setup_from_config


from time import time

import os
import sys

OUTPUT_DIR = "compile_test"


def timed_compile(model, output_dir: str, n_threads: int | None) -> float:
    """Convert `model` into a fresh `output_dir` and time a full compile."""
    hls = FlowHLS.convert(model, output_dir = output_dir)

    begin = time()
    hls.compile(n_threads = n_threads)
    return time() - begin


def main():
    if "XILINX_AP_INCLUDE" not in os.environ:
        sys.exit("Set XILINX_AP_INCLUDE to an HLS_arbitrary_Precision_Types "
                 "directory, or source the Vitis toolchain, to compile for emulation")

    config = FlatPuppiJetConfig(s1phi = False,               # Small model
                                n_steps = 4,                 # to speed up training
                                n_events = 100_000)
    config.tree_config["n_estimators"] = 20
    config.tree_config["max_depth"] = 2

    _, _, model, x, y = setup_from_config(config)

    model.fit(x, y)

    # print("Warmup...")
    # timed_compile(model, f"{OUTPUT_DIR}_warmup", n_threads = 1)

    # print("Compiling with a single thread...")
    # elapsed_single = timed_compile(model, f"{OUTPUT_DIR}_single", n_threads = 1)
    # print("Elapsed time: ", elapsed_single)

    print(f"Compiling with {os.cpu_count()} threads...")
    elapsed_multi = timed_compile(model, f"{OUTPUT_DIR}_multi", n_threads = None)
    print("Elapsed time: ", elapsed_multi)


if __name__ == "__main__":
    main()
