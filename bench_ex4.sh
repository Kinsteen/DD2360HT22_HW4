#!/bin/bash
i=32
while [[ $i -le 134217728 ]]
do
./lab4_ex4 $i 1000
i=$(($i*2))
done