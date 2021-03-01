import os
import Bio
from Bio.PDB import PDBParser, PDBIO, Select, NeighborSearch, Selection
#from Exception import LookupError

# https://stackoverflow.com/questions/61390035/how-to-save-each-ligand-from-a-pdb-file-separately-with-bio-pdb

def is_het(residue):
    res = residue.id[0]
    return res != " " and res != "W"

class LigandSelect(Select):
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
    """ Creates ligand file and protein (no water, etc.) file """
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
                    io.save(path_out + f"{pdb_code}_lig_{i}.pdb", LigandSelect(chain, residue))
                    i += 1
            if i > 1:
                print(f"Multiple ligands found for {pdb_code} ({i})!")
            
        