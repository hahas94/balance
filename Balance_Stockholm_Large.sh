#!/bin/sh
#SBATCH -t 14:00:00
#SBATCH -J Balance_Stockholm_Large
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -o "~/balance/results/stockholm_large/logs.txt"
#SBATCH -e "~/balance/results/stockholm_large/logs.txt"
#SBATCH --mail-user=hardy.hasan@liu.se
#SBATCH --mail-type=ALL

#sleep 100
module load gurobi/9.0.0-nsc1
module load Python/3.10.4-env-hpc1-gcc-2022a-eb
source venv/bin/activate

python3 main.py
