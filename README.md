# Simulated Molecular Evolution

## Overview
A lightweigth framework for simulated molecular evolution using:
- Genetic Algorithms (GA) with $\mu+\lambda$ replacement
- SELFIES-based molecular representation
- RDKit for molecular properties and MMFF94 energy
- Optional novelty-augmented fitness
- Interactive 2D molecular-soup visualization (pygame)

## Features
- SELFIES mutation and crossover operators
- Band-penalized fitness (energy, TPSA, logP, hetero penalties)
- Novelty scoring using Tanimoto distance
- Constraint filtering for chemical plausibility
- MVC architecture (Model: molecules & fitness, Controller: GA, View: visualization)
- Pygame real-time visualization with molecule selection & highlighting

## Installation

Requires **Python 3.10**

`pip install -r requirements.txt`

## Running a GA experiment (Console)
`python -m src.controller.experiment_runner --algo ga --gens 50`

Example with custom parameters:
`python -m src.controller.experiment_runner
--algo ga
--gens 100
--mu 40 --lam 40
-- mutation 0.4 --crossover 0.8`

## Running the Interactive Molecular Soup
`python -m src.view.run_soup`

Controls:
- SPACE - pause/resume
- UP/DOWN - speed control
- LEFT/RIGHT - step through generations
- Click molecule - inspect and show detailed view

## Project Structure
`
src/ 
    model/ # Molecule, fitness, constraints, operators, novelty 
    controller/ # GA pipeline, novelty search, experiment runner 
    view/ # Visualization (pygame), plotting utilities
`

## Authors

Anastasia Bertova

Matilde Pereira

Guillermo Torrealba

Bio-Inspired Artificial Intelligence 2025, UniTN
