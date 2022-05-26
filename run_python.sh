#!/bin/bash

i=$SLURM_ARRAY_TASK_ID
function=$1
params=$2

source activate audneurorl
module add cuda

echo "---------------------"
echo "launching "$function" with parameters "$params" run number "$i
python $function $params -run_num=$i || exit
echo "finished"