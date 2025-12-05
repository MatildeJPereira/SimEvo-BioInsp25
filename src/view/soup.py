import io

import pygame
import random
import numpy as np
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

    # def mol_to_surface(self, mol):
    #     img = MolToImage(mol.rdkit_mol, size=(120, 120))
    #     mode = img.mode
    #     size = img.size
    #     data = img.tobytes()
    #     return pygame.image.fromstring(data, size, mode).convert_alpha()

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

    def draw(self, screen):
        screen.blit(self.surface, (self.x, self.y))

class MolecularSoupPygame:
    def __init__(self, population, width=1000, height=800):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption('Molecular Soup')
        self.width = width
        self.height = height
        self.population = population
        self.sprites = self.create_sprites(population)

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

    def run(self, update_callback=None):
        running = True
        clock = pygame.time.Clock()
        frame = 0

        while running:
            clock.tick(60)
            frame += 1

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            for sprite in self.sprites:
                sprite.update(self.width, self.height)

            self.screen.fill((30, 20, 70))

            for sprite in self.sprites:
                sprite.draw(self.screen)

            pygame.display.flip()

            if update_callback and frame % 300 == 0:
                new_pop = update_callback(self.population)
                self.update_population(new_pop)

        pygame.quit()