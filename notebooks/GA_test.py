import sys
sys.path.append("../")

from src.controller.ga import GeneticAlgorithm, GAConfig
from src.model.molecule import Molecule
from src.model.population import Population
from src.model.fitness import compute_fitness_penalized

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

ga = GeneticAlgorithm(cfg, compute_fitness_penalized)

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

