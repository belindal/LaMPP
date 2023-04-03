#!/bin/bash

output_name=$1
shift
output_path="expts/crosstask_no-bkg/${output_name}"

mkdir -p $output_path

export PYTHONPATH="src/":$PYTHONPATH

python -u src/main.py \
        --dataset crosstask \
        --model_output_path $output_path \
        --remove_background \
        $@ \
        | tee ${output_path}/log.txt
