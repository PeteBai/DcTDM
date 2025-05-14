#!/bin/bash
#SBATCH --job-name=ddmxl_frames
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --mem=64GB
#SBATCH --time=23:59:59
#SBATCH --partition=short
#SBATCH --error=ddmxl_frames.%j.err

module load anaconda3/2022.05
source activate /work/zura-storage/Workspace/condaenv2/dgtdm
python /work/zura-storage/Workspace/DgTDM/dataset_scripts/gen_frames.py