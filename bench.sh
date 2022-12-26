#!/bin/bash
input="seg_sizes"
while IFS= read -r line
do
  ./lab4_ex2 4194304 $line
done < "$input"