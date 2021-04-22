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



def is_het(residue):
    '''
    returns false if it is a normal atom or a water molecule
    '''
    res = residue.id[0]
    return res != " " and res != "W"

class LigandSelect(Select):
    '''
    modify the Bio.PDB.Select class to filter only HETATMs
    https://stackoverflow.com/questions/61390035/how-to-save-each-ligand-from-a-pdb-file-separately-with-bio-pdb
    '''
    def __init__(self, chain, residue):
        self.chain = chain
        self.residue = residue

    def accept_chain(self, chain):
        return chain.id == self.chain.id

    def accept_residue(self, residue):
        """ Recognition of heteroatoms - Remove water molecules """
        return residue == self.residue and is_het(residue)
    
class ProteinSelect(Select):
    """
    Selects protein, no water and ligand
    """
    def accept_residue(self, residue):
        residues = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLU", "GLN", \
                "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", \
                "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
        if residue.get_resname() in residues:
            return True
        return False
    
def decouple_pdb(path_in="../pdb/train_coupled/", path_out="../pdb/train_decoupled/"):
    """ Creates ligand file and protein (no water, etc.) file 
    https://stackoverflow.com/questions/61390035/how-to-save-each-ligand-from-a-pdb-file-separately-with-bio-pdb
    """
    multi = 0
    for file in os.listdir(path_in):
        i = 0
        if file.endswith('.ent'):
            pdb_code = file[3:7]
            pdb = PDBParser().get_structure(pdb_code, path_in+file)
            io = PDBIO()
            io.set_structure(pdb)
            io.save(path_out+f"{pdb_code}_prot.pdb", ProteinSelect())
            for chain in pdb[0]:
                for residue in chain:
                    if not is_het(residue):
                        continue
                    if not (30<= len(residue) <= 66):
                        continue
                    #print(len(residue))
                    io.save(path_out + f"{pdb_code}_lig_{i}.pdb", LigandSelect(chain, residue))
                    i += 1
            if i > 1:
                #print(f"Multiple ligands found for {pdb_code} ({i})!")
                multi += 1
    print("total pdbs with multi", multi)

def LJ(epsilon, sigma, r):
    '''
    Returns 12-6 lennard jones
    epsilon: float, well depth
    sigma: float, bond radii
    r: float, distance of interaction
    return: float, LJ potential energy
    '''
    epsilon *= 3  # Huang et. Al. scales well-depth by 3 to preserve hydrophobic interactions
    return 4*epsilon*(((sigma/r)**12) - ((sigma/r)**6))

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
    
def p_bulk(pair, data):
    '''
    calcualtes the denominator of the g observed function
    '''
    V = (4*pi*(10**3))/3
    N =  sum(data[pair].values())
    return (1/M) * (1/V) * N

def p(pair, r, data):
    '''
    calculates the numerator of the g observed function
    '''
    return (1/M)*(data[pair][r]/(4*pi*(r**2)*dr))

# extract ligand as a separate pdb from complex pdb
# and save to directory train_decoupled/

print("extracting ligand...")
decouple_pdb("../pdb/train_coupled/", "../pdb/train_decoupled/")

print("creating mol2 representation...")
subprocess.call("./pdb2mol2.sh")

# count the occurences
print("calculating count dictionary...")
pairs = dict()  # key: tuple(prot_atom.id, lig_atom.id, val: dict{k:distance (r-dr) v: count}
dr = 0.1
distance_grid = np.arange(dr, 10+dr, dr)

parser = PDBParser()
pdb_codes = np.load("train_pdb_codes.npy")
for n, pdb_code in enumerate(pdb_codes):
    protein = parser.get_structure("protein", f"../pdb/train_decoupled/{pdb_code}_prot.pdb")[0]
    ligand_files = [os.path.basename(i) for i in glob.glob(f'../pdb/train_decoupled/{pdb_code}_lig_*.pdb')]
    ns = NeighborSearch(Selection.unfold_entities(protein, "A"))
    prot_mol2_map = dict()  # maps pdb to mol2 key: atom number, value: atom type
    lig_mol2_map = dict()
    
    # get mol2 map of prot
    with open(f"../pdb/train_decoupled_mol2/{pdb_code}_prot.mol2", "rb") as fp:
        for line in fp:
            line = line.decode('utf-8')
            if not len(line.split()):
                continue
            if line.split()[0] == '@<TRIPOS>ATOM':
                break
        
        for line in fp:
            line = line.decode("utf-8")
            if "@" in line.split()[0]:  # stop if next section in file
                break
            else:
                line = line.split()
                coord = np.around([float(line[2]), float(line[3]), float(line[4])], 3)
                prot_mol2_map[tuple(coord)] = line[5]
    # loop through ligands
    for ligand_file in ligand_files:
        
        # get mol2 map of lig
        with open(f"../pdb/train_decoupled_mol2/{ligand_file[:-4]}"+".mol2", "rb") as fp:
            for line in fp:
                line = line.decode('utf-8')
                if not len(line.split()):
                    continue
                if line.split()[0] == '@<TRIPOS>ATOM':
                    break
            
            for line in fp:
                line = line.decode("utf-8")
                if "@" in line.split()[0]:  # stop if next section in file
                    break
                else:
                    line = line.split()
                    coord = np.around([float(line[2]), float(line[3]), float(line[4])], 3)
                    lig_mol2_map[tuple(coord)] = line[5]
        
        # loop through ligand atoms
        ligand = parser.get_structure("ligand", f"../pdb/train_decoupled/{ligand_file}")[0]
        for lig_atom in ligand.get_atoms():
            lig_coord = tuple(np.around(np.float64(lig_atom.coord), 3))
            lig_id = lig_mol2_map[lig_coord]
            # go through radial spheres of 0.1 to 10 angstroms
            for r in distance_grid:
                r = round(r, 2)
                lower_bound = ns.search(lig_atom.coord, radius=r-(dr/2))
                upper_bound = ns.search(lig_atom.coord, radius=r+(dr/2))
                prot_atoms = list(set(upper_bound)-set(lower_bound))  # get neighboring protein atoms within the radial shell only
                if len(prot_atoms) > 0:
                    for prot_atom in prot_atoms:  # add each to dictionary
                        prot_coord = tuple(np.around(np.float64(prot_atom.coord), 3))
                        prot_id = prot_mol2_map[prot_coord]
                        if (prot_id, lig_id) not in pairs:
                            pairs[(prot_id, lig_id)] = dict()
                            pairs[(prot_id, lig_id)][r] = 1
                        else:
                            if round(r,2) not in pairs[(prot_id, lig_id)]:
                                pairs[(prot_id, lig_id)][r] = 1
                            else:
                                pairs[(prot_id, lig_id)][r] += 1

with open('../data/p_obs.pkl', 'wb') as handle:
    print("saving p_obs.pkl...")
    pickle.dump(pairs, handle, protocol=pickle.HIGHEST_PROTOCOL)


print("loading p_obs...")
p_obs = pickle.load(open("../data/p_obs.pkl", "rb"))  # the count of each atom at r distance dictionary
p_obs_trunc = {}  # remove <500 occurence

for pair in p_obs.keys():
    if sum(p_obs[pair].values()) > 500:
        p_obs_trunc[pair] = p_obs[pair]

dr = 0.1
g_obs = {}  # key: (prot_atom, lig_atom), val: dict{key: r, val: g_obs}
g_obs_avg = {}  # this averages similar atom types
M = len(p_obs_trunc.values())  # number of complexes
distance_grid = np.arange(dr, 10+dr, dr)
kB = 1 #0.001985875 #1.380649e-23
T = 1 #298

# creating g_obs
print("calculating g_obs...")
for pair in p_obs_trunc.keys():
    g_obs[pair] = dict()
    for r in distance_grid:
        r = round(r, 1)
        if r in p_obs_trunc[pair]:
            g_obs[pair][r] = p(pair, r, p_obs_trunc)/p_bulk(pair, p_obs_trunc)
        else:
            g_obs[pair][r] = 0

# creating PMF
print("calculating PMF...")
PMF = dict()  # key: tuple(prot_atom, lig_atom), val: np.array(-kBTlb(g_obs(r)))
for pair in g_obs.keys():
    y = g_obs[pair].values()
    PMF[pair] = np.array([-kB*T*np.log(i) if i != 0 else 3 for i in y])

with open("../data/g_obs.pkl", "wb") as fp:
    print("saving g_obs...")
    pickle.dump(g_obs, fp)

with open("../data/pmf.pkl", "wb") as fp:
    print("saving PMF...")
    pickle.dump(PMF, fp)

# epsilon and sigma values of each atom from AMBER parameters
# epsilon values are averaged by a geometric mean
# sigma values are averaged by an arithmetic mean
epsilon = {"O":0.2100, "C":0.0860, "N":0.1700, "S":0.2500, "P":0.2000, "F":0.061, "Cl":0.265, "Br":0.320, "I": 0.40}
vdw_radius = {"O":1.6612, "C":1.908, "N":1.8240, "S":2, "P": 2.1, "F": 1.75, "Cl": 1.948, "Br": 2.22, "I": 2.35}

print("initializing pair potentials...")
atom_pairs = np.stack(np.meshgrid(list(vdw_radius.keys()), list(vdw_radius.keys())), -1).reshape(-1, 2).tolist()  # all combinations of valid atom types
initial_pp = dict()

for pair in PMF.keys():
    pair_trunc = (pair[0].split(".")[0], pair[1].split(".")[0])  # remove hybridization information, tuple("C", "O")
    if list(pair_trunc) in atom_pairs:
        ui = []  # initial potential for atom pair
        ep_avg = np.sqrt(epsilon[pair_trunc[0]]*epsilon[pair_trunc[1]])  # geometric mean of epsilons of atom pair
        r_avg = (vdw_radius[pair_trunc[0]] + vdw_radius[pair_trunc[1]])/2  # arithmetic mean of vdw raii of atom pair
        for i, w in enumerate(PMF[pair]):
            distance = distance_grid[i]  # get distance of interaction
            v = LJ(ep_avg, r_avg, distance)
            ui.append((v*np.exp(-v) + w*np.exp(-w))/(np.exp(-v)+np.exp(-w)))
        initial_pp[pair] = ui
        print(f"{pair} initialized!")

with open("../data/init_pp.pkl", "wb") as fp:
    print("saving initial pair potentials...")
    pickle.dump(initial_pp, fp)

print("smoothing initial pair potentials...")

smooth = dict()  # smooth functions; key: pair, value: smooth array (y values)
parms = dict()  # contains the fitted params; key: pair, value: (vdw_parm, morse_parm)

for pair in initial_pp.keys():
    try:
        y = initial_pp[pair]  # initial potential
        pair_trunc = (pair[0].split(".")[0], pair[1].split(".")[0])  # remove hybridization information, tuple("C", "O")
        sigma_fit = (vdw_radius[pair_trunc[0]] + vdw_radius[pair_trunc[1]])/2
        epsilon_fit = np.sqrt(epsilon[pair_trunc[0]]*epsilon[pair_trunc[1]])
        r_min = (2**(1/6))*sigma_fit  # the radius at which potential is minimum
        rc = 6  # after 6 angstrom, potential energy = 0
        index_lt_3 = np.argmax(np.array(y)<3)
        index_gt_r_min = np.argmax(distance_grid>r_min)  # index of distance_grid greater than r_min
        index_gt_rc = np.argmax(distance_grid>rc)  # index of distance_grid greater than rc
        vdw_parm, vdw_pcov = curve_fit(vdw_repulsion, distance_grid[index_lt_3:index_gt_r_min], y[index_lt_3:index_gt_r_min])
        morse_parm, morse_pcov = curve_fit(morse, distance_grid[index_gt_r_min:index_gt_rc], y[index_gt_r_min:index_gt_rc])
        y_vdw = vdw_repulsion(distance_grid[index_lt_3:index_gt_r_min], *vdw_parm) # smooth vdw part
        y_morse = morse(distance_grid[index_gt_r_min:index_gt_rc], *morse_parm)  # smooth morse part
        y_0 = np.zeros(len(distance_grid[index_gt_rc:]))  # 0 after r > rc
        smooth[pair] = np.concatenate([y_vdw, y_morse, y_0])  # concatenate
        parms[pair] = (vdw_parm, morse_parm)
    except RuntimeError:
        print(f"ERROR {pair} could not be fitted!")

with open("../data/smooth.pkl", "wb") as fp:
    print("saving smooth initial pair potentials...")
    pickle.dump(smooth, fp)

with open("../data/fitted_parameters.pkl", "wb") as fp:
    print("saving fitted parameters...")
    pickle.dump(parms, fp)
