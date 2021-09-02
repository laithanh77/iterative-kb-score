# iterative-kb-score
iterative knowledge-based protein-ligand scoring function inspired from https://doi.org/10.1002/jcc.20504

for a class project. scoring function is partially implemented (the statistical potentials are crude)

download pdb files from rcsb website and unzip it into the pdb/train_coupled/ dir. It should be in the form of pdb{ID}.ent

requirements

1. conda installation: biopython, scipy, numpy

2. openbabel

usage:

download pdb files from rcsb website and unzip it into the pdb/train_coupled/ dir.
Go to src/
run "python program.py &> output.txt &"
wait 30 or so minutes
use the ipynb in data/ to view the results
