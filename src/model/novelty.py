# Computes novelty score using Tanimoto distance against an archive
# Manages an "archive" of diverse molecules

# # TODO trial version that can be changed
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

        mol.novelty = sum(distances[: self.k]) / self.k
        return mol.novelty

    def add(self, mol):
        self.archive.append(mol)

# from rdkit import Chem, DataStructs
# from rdkit.Chem import AllChem
#
# def morgan_fp(mol, radius=2, nBits=2048):
#     # expects a sanitized RDKit mol (Hs optional)
#     return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
#
# def tanimoto_distance_to_archive(fp, archive_fps):
#     if len(archive_fps) == 0:
#         return 1.0  # maximally novel if archive empty
#     sims = DataStructs.BulkTanimotoSimilarity(fp, archive_fps)  # list of sims
#     max_sim = max(sims)
#     return 1.0 - max_sim
#
# def compute_population_novelty(population):
#     """
#     Input: population = list of molecule objects (must have molecule.smiles)
#     Output: sets molecule.novelty for each molecule
#     """
#
#     # 1) Generate fingerprints
#     fp_dict = {}
#     for mol in population:
#         rd_mol = Chem.MolFromSmiles(mol.smiles)
#         if rd_mol is None:
#             fp_dict[mol] = None
#         else:
#             fp_dict[mol] = morgan_fp(rd_mol)
#
#     # 2) Compute novelty for each molecule
#     for mol, fp in fp_dict.items():
#         if fp is None:
#             mol.novelty = 1.0
#             continue
#         archive = [fp2 for m2, fp2 in fp_dict.items() if m2 != mol and fp2 is not None]
#         mol.novelty = tanimoto_distance_to_archive(fp, archive)












