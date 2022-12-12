#!/bin/bash

python3 src/run/run_models_across_time.py \
    --task_name "beta" \
    --task_phase "fit" \
    --test_split "Providence" \
    --test_dataset "all" \
    --context_width 0 \
    --use_tags "false" \
    --model_type "flat_unigram" \
    --training_split "no-split" \
    --training_dataset "no-dataset" \
    --examples_mode "true"