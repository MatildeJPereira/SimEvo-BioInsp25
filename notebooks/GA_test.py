import sys
sys.path.append("../")

from src.controller.ga import GeneticAlgorithm, GAConfig
from src.model.molecule import Molecule
from src.model.population import Population
from src.model.fitness import compute_fitness_penalized, novelty_augmented_fitness

# Phosphor molecule we can add back later '[O][P][=Branch1][C][=O][Branch1][C][O][O]'

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
params={'novelty_weight': 0.27431148086451107, 'w_energy': -0.4842840532722916, 'w_tpsa': -0.7246637144082346, 'w_logp': 0.9915391504042148, 'w_hetero': -0.8013546226403754}

#10 gen best
params={'novelty_weight': 0.6180132166303702, 'w_energy': -0.08873746080592704, 'w_tpsa': -0.777414081516488, 'w_logp': 0.6136156144531175, 'w_hetero': -0.3988390962845214}

#15 gen
#params={'novelty_weight': -0.4403734375876669, 'w_energy': 0.7948100567919711, 'w_tpsa': -0.7243509368688497, 'w_logp': -0.32908971279110144, 'w_hetero': -0.5910537976664327}

#20 gen 
#params={'novelty_weight': 0.47311668291256526, 'w_energy': 0.33287674537060297, 'w_tpsa': -0.8315825100999428, 'w_logp': -0.7378674342251692, 'w_hetero': 0.546824985633023}

#without novelty

#20 gen
params={'novelty_weight': 0, 'w_energy': 0.05985470110082258, 'w_tpsa': -0.9100448002447541, 'w_logp': 0.20528996961759582, 'w_hetero': -0.02761460203758359}

#10 gen
params={'novelty_weight': 0, 'w_energy': -0.03632714143203408, 'w_tpsa': -0.7203894704371934, 'w_logp': -0.8319049244822354, 'w_hetero': 0.6996763649393194}
params={'novelty_weight': 0, 'w_energy': -2.1554807472084873e-05, 'w_tpsa': -0.7636313151225684, 'w_logp': 0.39727110393992576, 'w_hetero': 0.06471711681256193}

#5 gen best
params={'novelty_weight': 0, 'w_energy': -0.1266398758035003, 'w_tpsa': 0.06247975459209559, 'w_logp': 0.17361012570960666, 'w_hetero': 0.09148137558222172}

ga = GeneticAlgorithm(cfg, novelty_augmented_fitness,**params)

history = ga.evolve(pop, generations=20)
history_pop= [x[0] for x in history]
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

from src.view.plots import plot_fitness_over_time, plot_all_atom_stats

plot_fitness_over_time(history_pop)
plot_all_atom_stats(history)

