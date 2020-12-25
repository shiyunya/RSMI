#!/bin/sh
for c in 100000 1000000 10000000
do
    for d in uniform normal skewed
    do
        if [ $d = "skewed" ]; then
            #echo "taskset -c 0,2,4,6,8 ./Exp -c $c -d $d -s 4"
            taskset -c 0,2,4,6,8 ./Exp -c $c -d $d -s 4 >> "$d"_"$c"_rsmi.out
            taskset -c 0,2,4,6,8 ./Exp -c $c -d $d -s 4  -z >> "$d"_"$c"_zm.out
        else
            #echo "taskset -c 0,2,4,6,8 ./Exp -c $c -d $d -s 1"
            taskset -c 0,2,4,6,8 ./Exp -c $c -d $d -s 1 >> "$d"_"$c"_rsmi.out
            taskset -c 0,2,4,6,8 ./Exp -c $c -d $d -s 1 -z >> "$d"_"$c"_zm.out
        fi
    done
done