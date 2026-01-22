#!/bin/sh
#
#SBATCH --output=slurm-LUnet-server-%x.%j.out
#SBATCH --mem 30GB      # memory pool for all cores
#SBATCH -n 5           # nb of threads
#SBATCH -t 2-00:00      # time (D-HH:MM)


Rscript LUnet-server4.r