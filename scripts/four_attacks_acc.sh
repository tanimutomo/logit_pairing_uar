#!/bin/bash

weight_path=$1
gpu_id=$2

# ./scripts/calc_acc.sh pgd_linf ${weight_path} ${gpu_id}
# ./scripts/calc_acc.sh pgd_l2 ${weight_path} ${gpu_id}
./scripts/calc_acc.sh fw_l1 ${weight_path} ${gpu_id}
./scripts/calc_acc.sh jpeg_linf ${weight_path} ${gpu_id}
./scripts/calc_acc.sh jpeg_l1 ${weight_path} ${gpu_id}
./scripts/calc_acc.sh elastic ${weight_path} ${gpu_id}
