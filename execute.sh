#!/bin/bash


for disk in `seq 1 8`; do
    make reverse DISK=$disk
done

mkdir -p results
mv fwd_disks_* results
mv rev_disks_* results

for disk in `seq 1 4`; do
    make reverse-mpi DISK=$disk
done

mkdir -p results-mpi
mv fwd_disks_* results-mpi
mv rev_disks_* results-mpi
