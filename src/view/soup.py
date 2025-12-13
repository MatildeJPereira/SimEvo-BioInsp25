import io

import pygame
import random
import numpy as np
from pygments import highlight
from rdkit.Chem.Draw import MolToImage, rdMolDraw2D
from io import BytesIO
import PIL.Image

class MolecularSprite:
    def __init__(self, molecule, x, y):
        self.molecule = molecule
        self.x = x
        self.y = y
        self.vx = random.uniform(-1, 1)
        self.vy = random.uniform(-1, 1)
        self.surface = self.mol_to_surface(molecule)

    def mol_to_surface(self, mol, size=(300, 300)):
        d2d = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
        d2d.drawOptions().clearBackground = False
        d2d.DrawMolecule(mol.rdkit_mol)
        d2d.FinishDrawing()

        png = d2d.GetDrawingText()
        bio = io.BytesIO(png)
        surface = pygame.image.load(bio, "temp.png").convert_alpha()
        surface = pygame.transform.smoothscale(surface, (130, 130))
        return surface

    def update(self, width, height):
        self.x += self.vx
        self.y += self.vy

        if self.x < 0 or self.x > width -120:
            self.vx *= -1
        if self.y < 0 or self.y > height -120:
            self.vy *= -1

    def draw(self, screen, highlight=False):
        if highlight:
            radius = int(self.surface.get_width() / 2)
            fade_circle = draw_fading_circle(radius, (255, 215, 100))
            screen.blit(fade_circle, (self.x, self.y))
        screen.blit(self.surface, (self.x, self.y))

def draw_fading_circle(radius, color):
    surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
    for r in range(radius, 0, -1):
        alpha = int(255 * (r / radius))  # stronger in center
        pygame.draw.circle(surf, (*color, alpha), (radius, radius), r)
    return surf

class MolecularSoupPygame:
    def __init__(self, population, width=1000, height=800):
        pygame.init()
        self.sidebar_width = 280
        self.screen = pygame.display.set_mode((width + self.sidebar_width, height))
        pygame.display.set_caption('Molecular Soup')
        self.width = width
        self.height = height
        self.population = population
        self.sprites = self.create_sprites(population)

        self.paused = False
        self.speed = 60
        self.history = [population]
        self.current_index = 0
        self.font = pygame.font.SysFont('Arial', 20)

        self.selected_molecule = None
        self.selected_zoom_surface = None

    def handle_click(self, pos):
        mx, my = pos
        if mx > self.width:
            return
        self.selected_molecule = None
        self.selected_zoom_surface = None

        for sprite in self.sprites:
            sx, sy = sprite.x, sprite.y
            sw, sh = sprite.surface.get_width(), sprite.surface.get_height()

            if sx <= mx <= sx + sw and sy <= my <= sy + sh:
                mol = sprite.molecule
                self.selected_molecule = mol
                d2d = rdMolDraw2D.MolDraw2DCairo(250, 250)
                d2d.drawOptions().clearBackground = True
                d2d.DrawMolecule(sprite.molecule.rdkit_mol)
                d2d.FinishDrawing()
                png = d2d.GetDrawingText()
                bio = io.BytesIO(png)
                self.selected_zoom_surface = pygame.image.load(bio, "selected.png").convert_alpha()
                break

    def draw_sidebar(self):
        x0 = self.width
        pygame.draw.rect(self.screen, (40, 40, 60), (x0, 0, self.sidebar_width, self.height))

        y = 20
        def write(text):
            nonlocal y
            surf = self.font.render(text, True, (230, 230, 255))
            self.screen.blit(surf, (x0 + 15, y))
            y += 28

        write(f"Generation: {self.current_index}")
        write(f"Population Size: {len(self.population.molecules)}")

        if self.selected_molecule is not None:
            mol = self.selected_molecule
            write("--- Selected ---")
            write(f"SMILES: {mol.smiles[:22]}...")
            fit = self.population.fitness.get(mol, None)
            if fit is not None:
                write(f"Fitness: {fit:.3f}")
            from src.model.fitness import archive
            nov = archive.novelty_score(mol)
            write(f"Novelty: {nov:.3f}")

            if self.selected_zoom_surface:
                self.screen.blit(self.selected_zoom_surface, (x0 + 15, y+ 10))

        else:
            if hasattr(self.population, 'fitness') and self.population.fitness:
                best = min(self.population.fitness, key=lambda m: self.population.fitness[m])
                bf = self.population.fitness[best]
                write(f"Best Fitness: {bf:.3f}")
                try:
                    from src.model.fitness import archive
                    nov = archive.novelty_score(best)
                    write(f"Novelty: {nov:.3f}")
                except Exception:
                    write(f"Novelty: -")

        controls_y = self.height - 150

        def write_bottom(text):
            nonlocal controls_y
            surf = self.font.render(text, True, (230, 230, 255))
            self.screen.blit(surf, (x0 + 15, controls_y))
            controls_y += 28

        write_bottom("Controls:")
        write_bottom("SPACE = Pause/Play")
        write_bottom("UP/DOWN = Speed +/-")
        write_bottom("RIGHT = Next Generation")
        write_bottom("LEFT = Previous Generation")

    def create_sprites(self, population):
        sprites = []
        for mol in population.molecules:
            x = random.randint(0, self.width -120)
            y = random.randint(0, self.height -120)
            sprites.append(MolecularSprite(mol, x, y))
        return sprites

    def update_population(self, population):
        self.population = population
        self.sprites = self.create_sprites(population)

    def handle_key(self, event, update_callback):
        if event.key == pygame.K_SPACE:
            self.paused = not self.paused

        elif event.key == pygame.K_UP:
            self.speed = min(240, self.speed + 10)

        elif event.key == pygame.K_DOWN:
            self.speed = max(10, self.speed - 10)

        elif event.key == pygame.K_RIGHT:
            if self.current_index < len(self.history) - 1:
                self.current_index += 1
                nex = self.history[self.current_index]
                self.update_population(nex)
            else:
                if update_callback:
                    new = update_callback(self.population)
                    self.history.append(new)
                    self.current_index = len(self.history) - 1
                    self.update_population(new)

        elif event.key == pygame.K_LEFT:
            if self.current_index > 0:
                self.current_index -= 1
                prev = self.history[self.current_index]
                self.update_population(prev)

    def run(self, update_callback=None):
        running = True
        clock = pygame.time.Clock()
        frame = 0

        while running:
            clock.tick(self.speed)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    self.handle_key(event, update_callback)
                if event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(event.pos)

            if not self.paused:
                for sprite in self.sprites:
                    sprite.update(self.width, self.height)

            self.screen.fill((110, 100, 180))
            self.draw_sidebar()

            best_mol = None
            if hasattr(self.population, 'fitness') and self.population.fitness:
                best_mol = min(self.population.fitness, key=lambda m: self.population.fitness[m])

            for sprite in self.sprites:
                hl = (sprite.molecule == best_mol)
                sprite.draw(self.screen, highlight=hl)

            pygame.display.flip()

            if not self.paused:
                frame += 1
                if update_callback and frame % 300 == 0:
                    new_pop = update_callback(self.population)
                    self.history.append(new_pop)
                    self.current_index = len(self.history) - 1
                    self.update_population(new_pop)

        pygame.quit()