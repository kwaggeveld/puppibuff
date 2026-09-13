from puppibuff.configs import FlatPuppiJetConfig
from puppibuff.hls import FlowHLS
from puppibuff.utils import output_dir

from pathlib import Path
from time import time
import os

def timed_compile(model, codec, project: Path, n_threads: int | None) -> float:
    """Convert and write `model` into a fresh `project` directory, then time the
    compile alone.
    """
    hls = FlowHLS.convert(model, output_dir = str(project))
    hls.write(codec)

    begin = time()
    hls.compile(n_threads = n_threads)
    return time() - begin

def main():
    outdir = output_dir(__file__)

    config = FlatPuppiJetConfig(n_steps = 4,
                                n_events = 100_000)
    config.tree_config["n_estimators"] = 20
    config.tree_config["max_depth"] = 2

    _, codec, model, x, y = config.setup()

    model.fit(x, y)

    print("Compiling with a single thread...")
    elapsed_single = timed_compile(model, codec, outdir / "single", n_threads = 1)
    print("Elapsed time: ", elapsed_single)

    print(f"Compiling with {os.cpu_count()} threads...")
    elapsed_multi = timed_compile(model, codec, outdir / "multi", n_threads = None)
    print("Elapsed time: ", elapsed_multi)


if __name__ == "__main__":
    main()
