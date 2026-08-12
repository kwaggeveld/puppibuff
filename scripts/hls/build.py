from puppibuff.utils import from_zip
from puppibuff.hls import FlowHLS

import sys

MODEL = "models/testing_small"

def main():                             # Pass directory for the HLS project
    if len(sys.argv) < 2:               # as argument 1, the archive as 2
        sys.exit(f"Usage: {sys.argv[0]} <outdir> [model]")

    output_dir = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) >= 3 else MODEL

    _, codec, model = from_zip(model_path)

    hls = FlowHLS.convert(model, output_dir = output_dir)
    hls.write(codec)
    hls.build()
    hls.write_payload()

if __name__ == "__main__":
    main()
