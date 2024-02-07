#!/bin/sh
#SBATCH -t 00:05:00
#SBATCH -J Balance_Stockholm_Medium
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -o results/stockholm_medium/logs.txt
#SBATCH -e results/stockholm_medium/logs.txt
#SBATCH --mail-user=hardy.hasan@liu.se
#SBATCH --mail-type=ALL

#sleep 100
module load gurobi/9.0.0-nsc1
module load Python/3.10.4-env-hpc1-gcc-2022a-eb
source venv/bin/activate

python main.py > results/stockholm_medium/logs.txt
