#!/bin/sh
#SBATCH -t 24:00:00
#SBATCH -J Balance_Stockholm_Large
#SBATCH -N 1
#SBATCH --mem=128000
#SBATCH --exclusive
#SBATCH --mail-user=hardy.hasan@liu.se
#SBATCH --mail-type=ALL

#sleep 100
module load gurobi/9.0.0-nsc1
module load Python/3.10.4-env-hpc1-gcc-2022a-eb
source venv/bin/activate

python3 main.py --graph_name=stockholm_large --run_name="$1" --num_intents="$1" > results/stockholm_large_"$1"/logs.txt
