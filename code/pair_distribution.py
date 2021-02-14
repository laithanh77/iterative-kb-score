import os
import glob
import numpy as np
import Bio
import pickle
from Bio.PDB import PDBParser, PDBIO, Select, NeighborSearch, Selection

pairs = dict()  # key: tuple(prot_atom.id, lig_atom.id, val: dict{k:distance (r-dr) v: count}
dr = 0.1
distance_grid = np.arange(dr, 10+dr, dr)

parser = PDBParser()
pdb_codes = np.load("train_pdb_codes.npy")
for pdb_code in pdb_codes:
    protein = parser.get_structure("protein", f"../pdb/train_decoupled/{pdb_code}_prot.pdb")[0]
    ligand_files = [os.path.basename(i) for i in glob.glob(f'../pdb/train_decoupled/{pdb_code}_lig_*.pdb')]
    ns = NeighborSearch(Selection.unfold_entities(protein, "A"))
    
    for ligand_file in ligand_files:
        ligand = parser.get_structure("ligand", f"../pdb/train_decoupled/{ligand_file}")[0]
        for lig_atom in ligand.get_atoms():
            lig_id = lig_atom.get_id()
            for r in distance_grid:
                r = round(r, 2)
                lower_bound = ns.search(lig_atom.coord, radius=r-(dr/2))
                upper_bound = ns.search(lig_atom.coord, radius=r+(dr/2))
                prot_atoms = list(set(upper_bound)-set(lower_bound))
                if len(prot_atoms) > 0:
                    for prot_atom in prot_atoms:
                        prot_id = prot_atom.get_id()
                        if (prot_id, lig_id) not in pairs:
                            pairs[(prot_id, lig_id)] = dict()
                            pairs[(prot_id, lig_id)][r] = 1
                        else:
                            if round(r,2) not in pairs[(prot_id, lig_id)]:
                                pairs[(prot_id, lig_id)][r] = 1
                            else:
                                pairs[(prot_id, lig_id)][r] += 1

with open('g_obs.pkl', 'wb') as handle:
    pickle.dump(pairs, handle, protocol=pickle.HIGHEST_PROTOCOL)