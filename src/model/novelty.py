# Computes novelty score using Tanimoto distance against an archive
# Manages an "archive" of diverse molecules

from rdkit.DataStructs import TanimotoSimilarity

class NoveltyArchive:
    def __init__(self, k=5):
        self.archive = []
        self.k = k

    def novelty_score(self, mol):
        if not self.archive:
            return 1.0

        fps = [m.compute_fingerprint() for m in self.archive]
        mol_fp = mol.compute_fingerprint()

        distances = [1 - TanimotoSimilarity(mol_fp, fp) for fp in fps]
        distances.sort()

        return sum(distances[: self.k]) / self.k

    def add(self, mol):
        self.archive.append(mol)
