#!/bin/bash
#SBATCH --job-name=ddmxl_depth
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --mem=64GB
#SBATCH --time=23:59:59
#SBATCH --partition=multigpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --error=ddmxl_depth.%j.err

module load anaconda3/2022.05 cuda/11.8
source activate /work/zura-storage/Workspace/condaenv2/dgtdm
cd /work/zura-storage/Workspace/DgTDM
python midas_pipeline/test_midas.py