#!/bin/sh
#
#SBATCH --output=slurm-spoton-%x.%j.out
#SBATCH --mem 90GB      # memory pool for all cores
#SBATCH -n 30           # nb of threads
#SBATCH -t 3-00:00      # time (D-HH:MM)


Rscript spoton5.r