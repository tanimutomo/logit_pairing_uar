#!/bin/bash

attack=$1
weight_path=$2
gpu_id=$3

cmd="pipenv run python src/test.py --dataset cifar10 --attack ${attack} --weight_path ${weight_path}"

if [ -n "${gpu_id}" ]; then
  cmd="${cmd} --gpu_ids ${gpu_id}"
fi

if [ "${attack}" = "pgd_linf" ]; then
  set_eps=(1 2 4 8 16 32)
elif [ "${attack}" = "pgd_l2" ]; then
  set_eps=(40 80 160 320 640 2560)
elif [ "${attack}" = "fw_l1" ]; then
  set_eps=(195 390 780 1560 6240 24960)
elif [ "${attack}" = "jpeg_linf" ]; then
  set_eps=(0.03125 0.0625 0.125 0.25 0.5 1)
elif [ "${attack}" = "jpeg_l1" ]; then
  set_eps=(2 8 64 256 512 1024)
elif [ "${attack}" = "elastic" ]; then
  set_eps=(0.125 0.25 0.5 1 2 8)
fi

for eps in "${set_eps[@]}"
do
  ${cmd} --eps ${eps}
done

echo "Finished evaluation"
