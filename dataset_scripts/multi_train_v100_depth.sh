#!/bin/bash
#SBATCH --job-name=dgtdm_multi_train
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --mem=64GB
#SBATCH --time=23:59:59
#SBATCH --partition=multigpu
#SBATCH --gres=gpu:v100-sxm2:4
#SBATCH --error=myjob_v100.%j.err

module load anaconda3/2022.05 cuda/11.8
source activate /work/zura-storage/Workspace/condaenv2/dgtdm
cd /work/zura-storage/Workspace/DgTDM
accelerate launch --config_file ./configs/multigpu.yaml train.py --config configs/makelongvideo_depth.yaml