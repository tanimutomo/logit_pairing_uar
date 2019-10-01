#!/bin/bash

weight_path=$1

# ./scripts/calc_acc.sh fw_l1 ${weight_path}
# ./scripts/calc_acc.sh jpeg_l2 ${weight_path}
./scripts/calc_acc.sh elastic ${weight_path}
./scripts/calc_acc.sh fog ${weight_path}
