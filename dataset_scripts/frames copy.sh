#!/bin/bash
#SBATCH --job-name=ddmxl_zip
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --mem=64GB
#SBATCH --time=23:59:59
#SBATCH --partition=short
#SBATCH --error=ddmxl_zip.%j.err

zip -r DSDDM.zip /work/zura-storage/Data/DrivingSceneDDM -x /work/zura-storage/Data/DSDDM_XL/source/**\*  