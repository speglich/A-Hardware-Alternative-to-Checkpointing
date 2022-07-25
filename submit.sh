#!/bin/bash

for repeat in `seq 0 2`; do
    sbatch -p cpulongb -A cenpes-lde -J AIO execute.sh
done
