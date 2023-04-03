#!/bin/bash

expt_folder=$1

line=$(grep "src/main.py" ${expt_folder}/log.txt | head -n1)

if [[ -z $line ]]
then
        echo "command not found in ${expt_folder}/log.txt"
        exit 1;
fi

decode_line=${line/model_output_path/model_input_path}

decode_line="$decode_line --force_optimal_assignment"

python -u $decode_line | tee ${expt_folder}/decode-optimal-assignment.out
