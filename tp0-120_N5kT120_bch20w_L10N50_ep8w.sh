#!/bin/bash

module load anaconda/2020.11
module load cuda/10.1
module load cudnn/7.6.5.32_cuda10.1

source activate tensorflow_37

export PYTHONUNBUFFERED=1

python tp0-120_N5kT120_bch20w_L10N50_ep8w.py