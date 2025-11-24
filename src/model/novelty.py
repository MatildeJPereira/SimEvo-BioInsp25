# Computes novelty score using Tanimoto distance against an archive
# Manages an "archive" of diverse molecules

# # TODO trial version that can be changed
# from rdkit.DataStructs import TanimotoSimilarity
#
# class NoveltyArchive:
#     def __init__(self, k=5):
#         self.archive = []
#         self.k = k
#
#     def novelty_score(self, mol):
#         if not self.archive:
#             return 1.0
#
#         fps = [m.compute_fingerprint() for m in self.archive]
#         mol_fp = mol.compute_fingerprint()
#
#         distances = [1 - TanimotoSimilarity(mol_fp, fp) for fp in fps]
#         distances.sort()
#
#         return sum(distances[: self.k]) / self.k
#
#     def add(self, mol):
#         self.archive.append(mol)

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

def morgan_fp(mol, radius=2, nBits=2048):
    # expects a sanitized RDKit mol (Hs optional)
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)

def tanimoto_distance_to_archive(fp, archive_fps):
    if len(archive_fps) == 0:
        return 1.0  # maximally novel if archive empty
    sims = DataStructs.BulkTanimotoSimilarity(fp, archive_fps)  # list of sims
    max_sim = max(sims)
    return 1.0 - max_sim 

def compute_novelty(smiles_list):
    """
    Returns: dict {smiles: novelty_score}
    Novelty = Tanimoto distance to most similar molecule in the rest of the list.
    """
    # 1) Build fingerprint dictionary
    fp_dict = {}
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        fp_dict[s] = morgan_fp(mol)

    # 2) Compute novelty for each molecule
    novelty_dict = {}
    for smi, fp in fp_dict.items():
        # archive = all fingerprints except this one
        archive = [fp2 for s2, fp2 in fp_dict.items() if s2 != smi]
        novelty = tanimoto_distance_to_archive(fp, archive)
        novelty_dict[smi] = novelty

    return novelty_dict

