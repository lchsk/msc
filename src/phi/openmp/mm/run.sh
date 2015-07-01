#!/bin/bash

# Default number of iterations
iter=10

app=$1

if [ -z $app ] || [ ! -f $app ]
    then
    echo "First parameter must be an executable file."
    exit 1
fi

if [ $2 ] 
    then
    iter=$2
fi

if [ $3 ]
    then
    export OMP_NUM_THREADS=$3
fi

echo "Running $app, $iter iteration(s), $3 thread(s)"

for i in `seq 1 $iter`;
do
  ./"$app"
done | grep "Time.*" | grep -o "[0-9.]\+" | awk 'BEGIN { s = 0 } { s = s + $0; n++ } END { printf "%.2f", s / n }'

echo