#!/usr/bin/env bash
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
ALL=(**blocks**)

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
    echo "[$(date +"%H:%M:%S")] Launched $b (pid $!)"
done

failed=()
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "--- ${names[$i]} done!"
    else
        echo "--- ${names[$i]} FAILED (see blocks/${names[$i]}/build.out)"
        failed+=("${names[$i]}")
    fi
done

if [ ${#failed[@]} -ne 0 ]; then
    echo "Failed: ${failed[*]}"
    exit 1
fi
echo "Built: ${blocks[*]}"
