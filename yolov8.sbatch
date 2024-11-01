#!/bin/bash

#SBATCH --job-name="YoloV10 ingredients detection"
#SBATCH --gpus=4
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --time=2-12:0 
#SBATCH --mail-user=aamotyakin@edu.hse.ru
#SBATCH --mail-type=ALL
#SBATCH --error="yolov10_"%j.err
#SBATCH --out="yolov10_"%j.out

module purge
module load Python

source deactivate
source activate yolov8_new

python yolo.py
