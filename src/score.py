import os
import Bio
from Bio.PDB import PDBParser, PDBIO, Select, NeighborSearch, Selection
import os
import glob
import numpy as np
import pickle
from math import pi
from scipy.optimize import curve_fit
import subprocess

def vdw_repulsion(distance, k, a, s):
    '''
    function to fit the repulsion part of the initial potentials
    k: float, fitting parameter
    a: float, fitting parameter
    s: float, fitting parameter
    '''
    global sigma_fit
    global epsilon_fit
    r_min = (2**(1/6))*sigma_fit # distance at minimum of potential
    return k * ((r_min - distance)**s) - epsilon_fit

def morse(distance, k, a, s):
    '''
    function to fit the morse potential part of the initial potentials
    k: float, fitting parameter
    a: float, fitting parameter
    s: float, fitting parameter
    '''
    global sigma_fit
    global epsilon_fit
    r_min = (2**(1/6))*sigma_fit # distance at minimum of potential
    return epsilon_fit * ((np.exp(-a*(distance-r_min))-1)**2) - epsilon_fit

id_list = ["3vhk", "4ag8", "4agc", "4asd", "4ase", "3wze"]  # test pdb ids
scoring_function = pickle.load(open("../data/smooth.pkl", "rb"))
parms = pickle.load(open("../data/fitted_parameters.pkl", "rb"))  # fitted parm dict: keys: pair values: (vdw_parm, morse_parm)
atom_pairs = list(scoring_function.keys())  # all possible atom pairs
epsilon = {"O":0.2100, "C":0.0860, "N":0.1700, "S":0.2500, "P":0.2000, "F":0.061, "Cl":0.265, "Br":0.320, "I": 0.40}
vdw_radius = {"O":1.6612, "C":1.908, "N":1.8240, "S":2, "P": 2.1, "F": 1.75, "Cl": 1.948, "Br": 2.22, "I": 2.35}


for pdb_id in id_list:
    score = 0  # the energy score
    parser = PDBParser()
    protein = parser.get_structure("protein", f"../pdb/score/pdb/{pdb_id}_prot.pdb")[0]
    ligand = parser.get_structure("ligand", f"../pdb/score/pdb/{pdb_id}_lig.pdb")[0]
    prot_mol2_map = dict()
    lig_mol2_map = dict()
    ns = NeighborSearch(Selection.unfold_entities(protein, "A"))
    # make prot mol2 map
    with open(f"../pdb/score/mol2/{pdb_id}_prot.mol2", "rb") as fp:
        for line in fp:
            line = line.decode('utf-8')
            if not len(line.split()):
                continue
            if line.split()[0] == '@<TRIPOS>ATOM':
                break
        
        for line in fp:
            line = line.decode("utf-8")
            if "@" in line.split()[0]:  # if it is the next section, leave
                break
            else:
                line = line.split()
                #print(line)
                coord = np.around([float(line[2]), float(line[3]), float(line[4])], 3)
                prot_mol2_map[tuple(coord)] = line[5]
                
    
    # make lig mol2 map
    with open(f"../pdb/score/mol2/{pdb_id}_lig.mol2", "rb") as fp:
        for line in fp:
            line = line.decode('utf-8')
            if not len(line.split()):
                continue
            if line.split()[0] == '@<TRIPOS>ATOM':
                break
        
        for line in fp:
            line = line.decode("utf-8")
            if "@" in line.split()[0]:  # if it is the next section, leave
                break
            else:
                line = line.split()
                coord = np.around([float(line[2]), float(line[3]), float(line[4])], 3)
                lig_mol2_map[tuple(coord)] = line[5]
    
    
    # calculating score
    
    for lig_atom in ligand.get_atoms():
        lig_coord = tuple(np.around(np.float64(lig_atom.coord), 3))
        lig_id = lig_mol2_map[lig_coord]
        neighbors = ns.search(lig_atom.coord, radius=10)  # protein atoms within 10 angstrom sphere
        for prot_atom in neighbors:
            prot_coord = tuple(np.around(np.float64(prot_atom.coord), 3))
            prot_id = prot_mol2_map[prot_coord]
            pair = (prot_id, lig_id)
            if pair in atom_pairs:
                pair_trunc = (pair[0].split(".")[0], pair[1].split(".")[0])  # remove hybridization information, tuple("C", "O")
                epsilon_fit = np.sqrt(epsilon[pair_trunc[0]]*epsilon[pair_trunc[1]])  # geometric mean of epsilons of atom pair
                sigma_fit = (vdw_radius[pair_trunc[0]] + vdw_radius[pair_trunc[1]])/2  # arithmetic mean of vdw raii of atom pair
                r_min = (2**(1/6))*sigma_fit  # the radius at which potential is minimum
                rc = 6  # after 6 angstrom, potential energy = 0
                distance = abs(lig_atom-prot_atom)
                if distance < r_min:
                    score += vdw_repulsion(distance, *parms[pair][0])
                elif r_min <= distance < rc:
                    score += morse(distance, *parms[pair][1])
    
    print(f"{pdb_id} COMPLEX SCORE IS {score}")
    
    
    
    