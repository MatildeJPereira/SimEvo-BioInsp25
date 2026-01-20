# Molecule class
# Handles SELFIES -> SMILES -> RDKit Mol conversion, descriptor extraction, MMFF94 energy computation,
# and cached Morgan fingerprints.

import selfies as sf
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, rdDistGeom, rdForceFieldHelpers, Crippen, rdFingerprintGenerator

class Molecule:
    """Represents a molecule encoded with SELFIES and its RDKit representation."""

    def __init__(self, selfies_str: str):
        self.selfies = selfies_str
        self.smiles = sf.decoder(selfies_str) # SELFIES -> SMILES
        self.rdkit_mol = Chem.MolFromSmiles(self.smiles) # SMILES -> RDKit

        # Basic descriptors
        self.heavy_atom_count = self.rdkit_mol.GetNumHeavyAtoms()
        self.tpsa = rdMolDescriptors.CalcTPSA(self.rdkit_mol)
        self.log_p = Crippen.MolLogP(self.rdkit_mol)
        self.num_carbons = self.count_carbons()

        # Cached values
        self.fingerprint = None
        self.energy = None # MMFF energy

    def compute_fingerprint(self):
        """Return or compute Morgan fingerprint (radius 2)."""
        if self.fingerprint is None:
            gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
            self.fingerprint = gen.GetFingerprint(self.rdkit_mol)
        return self.fingerprint

    def compute_mmff_energy(self):
        """Compute and store MMFF94 energy; return +inf on failure."""
        if self.energy is not None:
            return self.energy
        try:
            mol = Chem.AddHs(self.rdkit_mol)
            rdDistGeom.EmbedMolecule(mol)
            rdForceFieldHelpers.MMFFOptimizeMolecule(mol)
            props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol)
            ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(mol, props)
            self.energy = ff.CalcEnergy()
        except Exception:
            self.energy = float("inf") # invalid or optimization failed
        return self.energy

    def count_carbons(self):
        """Return the number of carbon atoms in an RDKit mol."""
        if self.rdkit_mol is None:
            return 0
        self.num_carbons = sum(1 for atom in self.rdkit_mol.GetAtoms() if atom.GetAtomicNum() == 6)
        return self.num_carbons
