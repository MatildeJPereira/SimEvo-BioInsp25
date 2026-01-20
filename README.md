# Simulated Molecular Evolution

## Overview
A lightweigth framework for simulated molecular evolution using:
- Genetic Algorithms (GA) with $\mu+\lambda$ replacement
- SELFIES-based molecular representation
- RDKit for molecular properties and MMFF94 energy
- Optional novelty-augmented fitness
- Interactive 2D molecular-soup visualization (pygame)

The project follows an MVC architecture:
- Model: molecules, fitness, constrains, operators, novelty archive
- Controller: Genetic Algorithm Implementation
- View: plotting utilities and pygame real-time visualization

Included are also Jupyter Notebooks for testing and output analysis. 

## Features
- SELFIES mutation (insert/delete/replace) and crossover operators
- Band-penalized fitness combining MMFF energy, TPSA, logP, hetero penalties
- Novelty scoring using Tanimoto distance over Morgan fingerprints
- Constraint filtering for chemical plausibility
- Interactive visualization with molecule selection, zoom view, best-fitness and best-novelty highlighting, 
pause/step/speed control

## Installation

Requires **Python 3.10**

`pip install -r requirements.txt`

## Running a GA experiment (Console)
`python -m src.controller.experiment_runner --gens 50`

Example with custom parameters:

`python -m src.controller.experiment_runner 
--gens 100 
--mu 40 --lam 40 
--mutation 0.4 --crossover 0.8`

The runner always uses the novelty-augmented penalized fitness.

## Running the Interactive Molecular Soup
`python -m src.view.run_soup`

Controls:
- SPACE - pause/resume
- UP/DOWN - speed control
- LEFT/RIGHT - step through generations
- Mouse click - select molecule and display details

The viewer highlights:
- Best-fitness molecule (yellow)
- Most-novel molecule (green)

## Project Structure

```graphql
notebooks/                      # Jupyter Notebooks folder
src/
    controller/
        experiment_runner.py    # CLI runner for GA experiments
        ga.py                   # Genetic Algorithm (μ + λ)
    model/
        constraints.py          # chemical plausibility filters
        fitness.py              # penalized fitness + novelty-augmented scoring
        molecule.py             # SELFIES molecules, RDKit conversion, descriptors
        novelty.py              # global novelty archive
        operators.py            # SELFIES mutation and crossover
        population.py           # population container + tournament selection
        stats.py                # statistical tests
    view/
        plots.py                # fitness-over-time and multi-run plots
        run_soup.py             # launches soup viewer
        soup.py                 # interactive 2D visualization
        viewer.py               # static molecule grid visualization
```

## Authors

- Anastasia Bertova
- Matilde Pereira
- Guillermo Torrealba

Bio-Inspired Artificial Intelligence, University of Trento 2025/2026
