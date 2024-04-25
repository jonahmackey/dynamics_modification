#!/bin/bash

out_dir="./slurm_out"

if [ ! -d "$out_dir" ]; then
    echo "Slurm output directory does not exist. Please create it."
    exit 1
fi

for file in slurm*; do
    if [ -f "$file" ]; then
        mv "$file" "$out_dir"
        echo "Moved $file to $out_dir"
    fi
done