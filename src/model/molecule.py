# Convert SELFIES to SMILES to RDKit molecules
# Validity Checks
# Caching fingerprints (for novelty calc)
# MMFF otpimization wrapper (energy computation)

# This is a trial version and can be changed later
import selfies as sf
from rdkit import Chem
from rdkit.Chem import AllChem

class Molecule:
    def __init__(self, selfies_str: str):
        self.selfies = selfies_str
        self.smiles = sf.decoder(selfies_str)
        self.rdkit_mol = Chem.MolFromSmiles(self.smiles)
        self.heavy_atom_count = self.rdkit_mol.GetNumHeavyAtoms()
        self.fingerprint = None
        self.energy = None