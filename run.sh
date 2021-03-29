#!/bin/bash
#SBATCH --account=def-m2nagapp
#SBATCH --cpus-per-task=6
#SBATCH --mem=6G
#SBATCH --time=0-24:00:00
#SBATCH --mail-user=hosseinkeshavarz1997@gmail.com
#SBATCH --mail-type=ALL

source venv/bin/activate
which python
python -u src/models.py
