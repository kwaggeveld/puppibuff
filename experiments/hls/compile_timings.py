from puppibuff.configs import FlatPuppiJetConfig
from puppibuff.hls import FlowHLS

from time import time
import os

OUTPUT_DIR = "compile_timings"

def timed_compile(model, output_dir: str, n_threads: int | None) -> float:
    """Convert `model` into a fresh `output_dir` and time a full compile."""
    hls = FlowHLS.convert(model, output_dir = output_dir)

    begin = time()
    hls.compile(n_threads = n_threads)
    return time() - begin

def main():
    config = FlatPuppiJetConfig(s1phi = False,
                                n_steps = 4,
                                n_events = 100_000)
    config.tree_config["n_estimators"] = 20
    config.tree_config["max_depth"] = 2

    _, _, model, x, y = config.setup()

    model.fit(x, y)

    print("Compiling with a single thread...")
    elapsed_single = timed_compile(model, f"{OUTPUT_DIR}_single", n_threads = 1)
    print("Elapsed time: ", elapsed_single)

    print(f"Compiling with {os.cpu_count()} threads...")
    elapsed_multi = timed_compile(model, f"{OUTPUT_DIR}_multi", n_threads = None)
    print("Elapsed time: ", elapsed_multi)


if __name__ == "__main__":
    main()
