# Population class storing molecules, their fitness, and metadata
# Utility methods for selection (tournament, roulette, novelty-based…)

# TODO Trial version that can be updated
import random
from rdkit.Chem import DataStructs, rdMolDescriptors
from rdkit.DataStructs import TanimotoSimilarity
import numpy as np
import selfies as sf
from rdkit import Chem

class ValidationSet:
    def __init__(self, mol_list):
        self.smiles = mol_list
        self.rdkit_mol = [Chem.MolFromSmiles(s) for s in self.smiles]
        self.fingerprints = [rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol, radius=2, nBits=2048
            ) if mol is not None else None for mol in self.rdkit_mol]

    def delete_similar_mol(self):
        delete_list=set()
        for i in range(len(self.fingerprints)):
            if self.fingerprints[i] is None:
                delete_list.add(i)
                continue
            if i in delete_list:
                continue
            for j in range(i+1,len(self.fingerprints)):
                if self.fingerprints[j] is None:
                    continue
                d=TanimotoSimilarity(self.fingerprints[i], self.fingerprints[j])
                if d>=0.5:
                    delete_list.add(j)
        for i in sorted(delete_list, reverse=True):
            self.smiles.pop(i)
            self.rdkit_mol.pop(i)
            self.fingerprints.pop(i)
class Population:
    def __init__(self, molecules):
        self.molecules = molecules
        self.fitness = {}

    def evaluate(self, fitness_fn):
        for mol in self.molecules:
            self.fitness[mol] = fitness_fn(mol)

    def select_tournament(self, k=3):
        candidates = random.sample(self.molecules, k)
        return min(candidates, key=lambda m: self.fitness[m])
    def compute_carbon_avg(self):
        n_carb=0
        for mol in self.molecules:
            n_carb+=mol.num_carbons
        avg=n_carb/len(self.molecules)
        return avg
    def compute_other_atoms_avg(self):
        n_atoms=0
        for mol in self.molecules:
            n_atoms+=mol.heavy_atom_count-mol.num_carbons
        avg=n_atoms/len(self.molecules)
        return avg
    def compute_carbon_pctg_avg(self):
        n_atoms=0
        for mol in self.molecules:
            n_atoms+=mol.num_carbons/mol.heavy_atom_count
        avg=n_atoms/len(self.molecules)
        return avg
    
    def compute_complexity_avg(self):
        compx=0
        for mol in self.molecules:
            compx+=mol.complexity
        avg=compx/len(self.molecules)
        return avg

    def compute_validation_knn_distance(self, validation_molecules, k=5):
        """
        Compute average KNN distance between validation molecules and
        the current population using each molecule's own fingerprint.

        validation_molecules: list of Molecule objects
        """

        # Precompute fingerprints for population
        pop_fps = [
            mol.compute_fingerprint() if mol.fingerprint is None else mol.fingerprint
            for mol in self.molecules
        ]


        if len(pop_fps) == 0:
            return None

        all_target_distances = []
        validation_molecules=ValidationSet(validation_molecules)


        for target in validation_molecules.fingerprints:

            # Similarities to population members
            if target is None:
                continue
            sims = [DataStructs.TanimotoSimilarity(target, fp) for fp in pop_fps if fp is not None]

            # Convert to distances
            dists = [1 - s for s in sims]

            # Pick k nearest neighbors
            knn = sorted(dists)[:k]

            all_target_distances.append(np.mean(knn))

        if not all_target_distances:
            return None

        return float(np.mean(all_target_distances))


