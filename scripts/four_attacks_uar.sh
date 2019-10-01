#!/bin/bash

weight_path=$1

./scripts/calc_uar.sh fw_l1 ${weight_path}
./scripts/calc_uar.sh jpeg_l2 ${weight_path}
./scripts/calc_uar.sh elastic ${weight_path}
./scripts/calc_uar.sh fog ${weight_path}
