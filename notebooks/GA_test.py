import sys
sys.path.append("../")

from src.controller.ga import GeneticAlgorithm, GAConfig
from src.model.molecule import Molecule
from src.model.population import Population
from src.model.fitness import compute_fitness_penalized, novelty_augmented_fitness

# Phosphor molecule we can add back later '[O][P][=Branch1][C][=O][Branch1][C][O][O]'
VALIDATION_GROUPS = {
    "alpha_amino_acids": [
            "NCC(=O)O",
    "NC(C)C(=O)O",
    "NCC(O)C(=O)O",
    "NC(CC(=O)O)C(=O)O",
    "NC(CCC(=O)O)C(=O)O",
    "NC(C(C)C)C(=O)O",
    "NCC(CO)C(=O)O",
    "N1CCCC1C(=O)O",
    "NC(Cc1ccccc1)C(=O)O",
    "NC(CS)C(=O)O",
    "NC(C=O)C(=O)O",
    "NC(CN)C(=O)O",
    "NCC(C)C(=O)O",
    "NC(CO)C(=O)O",
    "NC(Cc1[nH]cnc1)C(=O)O",
    "NC(Cc1ccc(O)cc1)C(=O)O",
    "NC(Cc1c[nH]c2ccccc12)C(=O)O",
    "NC(Cc1ccc(CO)cc1)C(=O)O",
    "NC(COO)C(=O)O",
    "NC(CCO)C(=O)O"
    ],

    "beta_gamma_amino_acids": [
            "NCCC(=O)O",
    "NCCCC(=O)O",
    "NCC(O)C(=O)O",
    "NCCCO",
    "NCCC(O)C(=O)O"
    ],

    "hydroxy_acids": [
    "CC(O)C(=O)O",        # lactic acid
    "OCC(=O)O",           # glycolic acid
    "OCC(O)C(=O)O",       # malic acid
    "OC(=O)CC(=O)O",      # succinic acid
    "O=C(O)CCC(=O)O"     # 4-hydroxybutyrate precursor
    ],

    "keto_acids": [
    "CC(=O)C(=O)O",       # pyruvate
    "O=C(O)C(=O)O",       # oxalate
    "O=C(O)CC(=O)O",      # malonate
    "O=C(O)CCC(=O)O",     # succinate
    "O=C(O)C(=O)C(=O)O",  # oxalosuccinate-like
    ],

    "dipeptides": [
    "NCC(=O)NCC(=O)O",        # Gly-Gly
    "NC(C)C(=O)NCC(=O)O",     # Ala-Gly
    "NCC(=O)NC(C)C(=O)O",     # Gly-Ala
    "NCC(=O)NC(CO)C(=O)O",    # Gly-Ser
    ],

    "polyamines": [
    "NCCN",
    "NCCCCN",
    "NCCCN",
    "NCCNCCN",
    "NCCCCCCN",
    "NCCCNCCN",
    ],

    "Small prebiotic N-containing molecules": [
    "NC=O",          # formamide
    "NC#N",          # cyanamide
    "N=C=O",         # isocyanic acid
    "N=C(N)N",       # diaminocarbene precursor
    "CNC=O"         # methylformamide
    ],

    "Nucleobase-like fragments": [
    "O=C1NC=NC=N1",            # uracil-like core
    "NC1=NC=NC=N1",            # adenine fragment
    "N1C=NC=N1",               # diaminopyrimidine
    "O=CNC=N",                 # formamidine-urea
    "O=C1NCCN1"               # imidazolidone
    ],

    "TCA/glycolysis intermediates": [
    "O=CC(O)C(=O)O",           # glycerate
    "O=CC(=O)CO",              # glyoxylate
    "O=CC(O)CO",               # glyceraldehyde
    "OC(CO)C(=O)O"            # hydroxypropionate
    ],

    "Cofactor/fragments": [
    "NC(=O)c1ccccn1",          # nicotinamide fragment
    "O=C(O)c1ccccn1",          # pyridinecarboxylate
    "c1ncc[nH]1",              # pyrimidine fragment
    "OC[C@H](O)[C@H](O)CO",    # sugar alcohol fragment (glycerol aldehyde-like)
    "O=C(O)c1ccncc1",          # pyridyl-carboxylate
    ],
}
soup = ['[C][#N]', '[C][=O]', '[C][O]', '[C][C][O]', '[C][C][=O]', '[O][=C][C][O]', '[O][=C][O]', '[N][C][=Branch1][C][=O][N]', '[N]', '[O]', '[N][C][C][=Branch1][C][=O][O]', '[C][C][Branch1][=Branch1][C][=Branch1][C][=O][O][N]', '[C][C][=Branch1][C][=O][O]', '[C][C][N]', '[C][S]', '[C][C][=Branch1][C][=O][C][=Branch1][C][=O][O]', '[C][C][=Branch1][C][=O][C]', '[O][=C][=O]', '[O][=C][=S]', '[O][P][=Branch1][C][=O][Branch1][C][O][O]', '[C][=C][C][=C][C][=C][Ring1][=Branch1]', '[C][=C][N][=C][NH1][Ring1][Branch1]', '[C][C][=C][NH1][C][=Ring1][Branch1]', '[C][C][C][C][C][Ring1][Branch1]', '[C][C][C][C][C][C][Ring1][=Branch1]']

initial = []
for s in soup:
    initial.append(Molecule(s))

pop = Population(initial)

cfg = GAConfig(
    mu=20,
    lam=40,
    mutation_rate=0.5,
    crossover_rate=0.8,
    tournament_k=2,
    random_seed=0
)

#params={'novelty_weight': -70.9891509038727, 'w_energy': -53.90979453309572, 'w_tpsa': 7.826734928172879, 'w_logp': -40.023769229018136, 'w_hetero': -50.506567471054844}
#params={'novelty_weight': 32.146497894175155, 'w_energy': -89.70640677859367, 'w_tpsa': -5.75288816540427, 'w_logp': -67.33963007591916, 'w_hetero': 22.691843955559293}

# with novelty

#5 gen
#params={'novelty_weight': 0.27431148086451107, 'w_energy': -0.4842840532722916, 'w_tpsa': -0.7246637144082346, 'w_logp': 0.9915391504042148, 'w_hetero': -0.8013546226403754}

#10 gen best
#params={'novelty_weight': 0.6180132166303702, 'w_energy': -0.08873746080592704, 'w_tpsa': -0.777414081516488, 'w_logp': 0.6136156144531175, 'w_hetero': -0.3988390962845214}
##params={'novelty_weight': -0.9496191486930635, 'w_energy': 0.03070839138899563, 'w_tpsa': -0.7723571017754911, 'w_logp': -0.8509744316164835, 'w_hetero': 0.3625192484189641}
params={'novelty_weight': 1.1285418704644736, 'w_energy': -0.33289083386978996, 'w_tpsa': -0.8828075895967606, 'w_logp': 0.437344545570551, 'w_hetero': 0.21209405936427506}

#15 gen
#params={'novelty_weight': -0.4403734375876669, 'w_energy': 0.7948100567919711, 'w_tpsa': -0.7243509368688497, 'w_logp': -0.32908971279110144, 'w_hetero': -0.5910537976664327}

#20 gen 
#params={'novelty_weight': 0.47311668291256526, 'w_energy': 0.33287674537060297, 'w_tpsa': -0.8315825100999428, 'w_logp': -0.7378674342251692, 'w_hetero': 0.546824985633023}

#without novelty

#params={'novelty_weight': 0, 'w_energy': -2.1554807472084873e-05, 'w_tpsa': -0.7636313151225684, 'w_logp': 0.39727110393992576, 'w_hetero': 0.06471711681256193}

ga = GeneticAlgorithm(cfg, novelty_augmented_fitness,**params)

history = ga.evolve(pop, generations=35,validation_groups=VALIDATION_GROUPS)
history_pop= [x['population'] for x in history]
print("Evolution done!")

for gen, p in enumerate(history_pop):
    print(f"\nGeneration {gen}")
    for n in p.molecules:
        print(n.smiles)

from src.view.viewer import population_grid

from PIL import Image, ImageDraw, ImageFont

def label_frame(img, gen):
    """Adds a generation label on top of the RDKit image."""
    img = img.copy()
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 100)
    except:
        font = ImageFont.load_default()

    label = f"Generation {gen}"
    draw.text((20, 20), label, fill=(0, 0, 0), font=font)

    return img


frames = []

for gen, p in enumerate(history_pop):
    print(f"Generation {gen}")
    img = population_grid(p, n=10)  # RDKit-generated image
    img = label_frame(img, gen)     # Add label
    frames.append(img)

# Save as GIF
frames[0].save(
    "evolution.gif",
    save_all=True,
    append_images=frames[1:], 
    duration=600,
    loop=0
)

from src.view.plots import plot_fitness_over_time, plot_all_atom_stats, plot_validation_by_group

plot_fitness_over_time(history_pop)
plot_all_atom_stats(history)
plot_validation_by_group(history)

