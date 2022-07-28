#!/bin/bash

echo $HOSTNAME

for repeat in `seq 0 0`; do
    time make
    #rm -rf /scr01/test.data
done
