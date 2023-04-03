#!/bin/bash

output_name=$1
shift
output_path="expts/crosstask_i3d-resnet/${output_name}"

mkdir -p $output_path

export PYTHONPATH="src/":$PYTHONPATH

python -u src/main.py \
    --dataset crosstask \
    --crosstask_feature_groups i3d resnet \
    --model_output_path $output_path \
    $@ \
    | tee ${output_path}/log.txt
