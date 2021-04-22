#!/bin/bash

for file in ../pdb/train_decoupled/*; do
	filename=$(basename -- "$file")
	extension="${filename##*.}"
	filename="${filename%.*}"
	obabel -i pdb $file -o mol2 -O ../pdb/train_decoupled_mol2/${filename}.mol2
done
