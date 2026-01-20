# Novelty scoring implementation using Tanimoto fingerprint distance.
# Stores an archive of previously discovered molecules and computes novelty as the mean k-nearest-neighbor distance.

from rdkit.DataStructs import TanimotoSimilarity

class NoveltyArchive:
    """Maintains a fingerprint archive and computes novelty scores."""

    def __init__(self, k=5):
        self.archive = []   # stored Molecule objects
        self.k = k          # number of neighbours to average

    def novelty_score(self, mol):
        """
        Return novelty as mean k-NN Tanimoto distance (1 - similarity).
        Higher novelty -> more structurally distinct from archive.
        """
        if not self.archive:
            return 1.0

        fps = [m.compute_fingerprint() for m in self.archive]
        mol_fp = mol.compute_fingerprint()

        distances = [1 - TanimotoSimilarity(mol_fp, fp) for fp in fps]
        distances.sort()

        return sum(distances[: self.k]) / self.k

    def add(self, mol):
        """Add a molecule to the novelty archive."""
        self.archive.append(mol)
