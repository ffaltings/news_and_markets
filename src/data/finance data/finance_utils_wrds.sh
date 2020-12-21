#!/bin/bash
#$ -cwd
#$ -pe onenode 8
#$ -l m_mem_free=6G
python3 finance_utils.py