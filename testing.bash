#!/bin/bash

# Define the number of times you want to run the command
num_times=5

# Loop to run the command multiple times
for ((i=1; i<=$num_times; i++)); do
    echo "Iteration $i:"
    python test_cost.py | grep "Cost route total time"
    echo "----------------------------------------"
done
