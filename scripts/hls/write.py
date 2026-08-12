from puppibuff.utils import from_zip
from puppibuff.hls import FlowHLS

import sys

MERGED = True

def main():                             # Pass directory for the HLS project
    if len(sys.argv) >= 2:              # as argument 1
        output_dir = sys.argv[1]
    else:
        output_dir = "flowhls"

    _, codec, model = from_zip("models/testing_small")
    
    hls = FlowHLS.convert(model, output_dir = output_dir, merged = MERGED)
    hls.write(codec)

if __name__ == "__main__":
    main()
