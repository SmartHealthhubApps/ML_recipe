#!/bin/bash

#SBATCH --job-name="Cuda test"
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=0-1:0 
#SBATCH --mail-user=aamotyakin@edu.hse.ru
#SBATCH --mail-type=ALL


module purge
module load Python

source deactivate
source activate yolov8_new

#python yolo.py

nvidia-smi	
which python
python -V
# conda list -e
python -c 'import torch; print(torch.cuda.is_available())'
