#!/usr/bin/env bash
OUTPUT_DIR="test_run"
NUM_PARALLEL_PROCESS=4
mkdir -p $OUTPUT_DIR

python process_vid.py  videos_taa/FovRend_None.mp4 $OUTPUT_DIR
