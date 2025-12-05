# Convert SELFIES to SMILES to RDKit molecules
# Validity Checks
# Caching fingerprints (for novelty calc)
# MMFF optimization wrapper (energy computation)

# This is a trial version and can be changed later
import selfies as sf
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, rdDistGeom, rdForceFieldHelpers, Crippen, rdFingerprintGenerator

class Molecule:
    def __init__(self, selfies_str: str):
        self.selfies = selfies_str
        self.smiles = sf.decoder(selfies_str)
        self.rdkit_mol = Chem.MolFromSmiles(self.smiles)
        self.heavy_atom_count = self.rdkit_mol.GetNumHeavyAtoms()
        self.novelty = None
        self.mmff_energy = None
        self.fingerprint = None
        self.energy = None
        self.tpsa = rdMolDescriptors.CalcTPSA(self.rdkit_mol)
        self.log_p = Crippen.MolLogP(self.rdkit_mol)
        self.num_carbons = self.count_carbons()

    def compute_fingerprint(self):
        if self.fingerprint is None:
            gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
            self.fingerprint = gen.GetFingerprint(self.rdkit_mol)
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

    def count_carbons(self):
        """Return the number of carbon atoms in an RDKit mol."""
        if self.rdkit_mol is None:
            return 0
        self.num_carbons = sum(1 for atom in self.rdkit_mol.GetAtoms() if atom.GetAtomicNum() == 6)
        return self.num_carbons # TODO Verify
