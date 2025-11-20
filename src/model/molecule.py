# Convert SELFIES to SMILES to RDKit molecules
# Validity Checks
# Caching fingerprints (for novelty calc)
# MMFF otpimization wrapper (energy computation)

# TODO This is a trial version and can be changed later
import selfies as sf
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, rdDistGeom, rdForceFieldHelpers

class Molecule:
    def __init__(self, selfies_str: str):
        self.selfies = selfies_str
        self.smiles = sf.decoder(selfies_str)
        self.rdkit_mol = Chem.MolFromSmiles(self.smiles)
        self.fingerprint = None
        self.energy = None

    def compute_fingerprint(self):
        if self.fingerprint is None:
            self.fingerprint = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                self.rdkit_mol, radius=2, nBits=2048
            )
        return self.fingerprint

    def compute_mmff_energy(self):
        if self.energy is not None:
            return self.energy
        try:
            mol = Chem.AddHs(self.rdkit_mol)
            rdDistGeom.EmbedMolecule(mol)
            result = rdForceFieldHelpers.MMFFOptimizeMolecule(mol)
            props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol)
            ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(mol, props)
            self.energy = ff.CalcEnergy()
        except Exception:
            self.energy = float("inf")
        return self.energy