#!/bin/bash
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mail-type=END

module add cuda/9.0
module add cudnn/7-cuda-9.0

python3 main.py
