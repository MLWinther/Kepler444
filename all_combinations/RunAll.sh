#!/bin/bash

N=8

for j in $(seq -f "%03g" 1 34); do 
   ((i=i%N)); ((i++==0)) && wait
   BASTArun --seed 444 input_$j.xml &
done

