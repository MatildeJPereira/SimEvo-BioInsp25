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
        self.novelty = None
        self.mmff_energy = None
        self.logp = None
        self.tpsa = None
        self.fingerprint = None
        self.fitness = None
        self.num_carbons = None

    def count_carbons(self):
        """Return the number of carbon atoms in an RDKit mol."""
        if self.rdkit_mol is None:
            return 0
        self.num_carbons = sum(1 for atom in self.rdkit_mol.GetAtoms() if atom.GetAtomicNum() == 6)