#!/bin/bash

module load python/3.9.6
module load gcc/11.1.0

echo $HOSTNAME

for repeat in `seq 0 0`; do
    time make
    #rm -rf /scr01/test.data
done
