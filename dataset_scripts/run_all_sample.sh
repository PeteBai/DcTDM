#!/bin/bash
#SBATCH --job-name=dgtdm_run_all_sample
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --mem=64GB
#SBATCH --time=23:59:59
#SBATCH --partition=multigpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --error=runall_v100.%j.err

module load anaconda3/2022.05 cuda/11.8
source activate /work/zura-storage/Workspace/condaenv2/dgtdm
cd /work/zura-storage/Workspace/DgTDM
python infer.py --width 256 --height 256 --config_file ./configs/makeicraexp.txt --prompt "A vehicle driving in-road in a small town with a river and several bridges; rainly, cloudy town surroundings." --depth_start_file /work/zura-storage/Data/DrivingSceneDDM/depth/carla_01/000000.npy --n_frames 96 --out_folder icraexp --do_gif --do_depth --do_canny --canny_start_file /work/zura-storage/Data/DrivingSceneDDM/canny/carla_01/000000.png --my_model_path ./outputs/slam_synthetic2/checkpoint-2326