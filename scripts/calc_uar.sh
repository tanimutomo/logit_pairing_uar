#!/bin/bash

attack=$1
weight_path=$2

eps_pgd_linf=(1 2 4 8 16 32)
eps_pgd_l2=(150 300 600 1200 2400 4800)
eps_fw_l1=(9562.5 19125 76500 153000 306000 612000)
eps_jpeg_linf=(0.062 0.125 0.250 0.500 1 2)
eps_jpeg_l2=(8 16 32 64 128 256)
eps_jpeg_l1=(256 1024 4096 16384 65536 131072)
eps_elastic=(0.250 0.500 2 4 8 16)
eps_fog=(128 256 512 2048 4096 8192)
eps_snow=(0.062 0.125 0.250 2 4 8)
eps_gabor=(6.250 12.500 25 400 800 1600)

if [ "${attack}" = "pgd_linf" ]; then
  set_eps=${eps_pgd_linf}
elif [ "${attack}" = "pgd_l2" ]; then
  set_eps=${eps_pgd_l2}
elif [ "${attack}" = "fw_l1" ]; then
  set_eps=${eps_fw_l1}
elif [ "${attack}" = "jpeg_linf" ]; then
  set_eps=${eps_jpeg_linf}
elif [ "${attack}" = "jpeg_l2" ]; then
  set_eps=${eps_jpeg_l2}
elif [ "${attack}" = "jpeg_l1" ]; then
  set_eps=${eps_jpeg_l1}
elif [ "${attack}" = "elastic" ]; then
  set_eps=${eps_elastic}
elif [ "${attack}" = "fog" ]; then
  set_eps=${eps_fog}
elif [ "${attack}" = "snow" ]; then
  set_eps=${eps_snow}
elif [ "${attack}" = "gabor" ]; then
  set_eps=${eps_gabor}
fi

for eps in "${set_eps[@]}"
do
  pipenv run python src/test.py --dataset cifar10 --attack ${attack} --eps ${eps} --weight_path ${weight_path}
done

echo "Finished evaluation"
