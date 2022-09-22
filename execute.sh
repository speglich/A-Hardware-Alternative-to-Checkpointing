#!/bin/bash

for disk in `seq 1 8`; do
    make DISK=$disk
    for rate in 2 4 8 16; do
        make compression DISK=$disk RATE=$rate
    done
done
