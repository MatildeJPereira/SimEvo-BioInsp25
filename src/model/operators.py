# Mutation & crossover on SELFIES: 
# - mutation: substitute, insert, delete 
# crossover: cut SELFIES at random position
# occasional sanity check by converting back to RDKit

# TODO This is a trial version and can be changed later
import random
import selfies as sf

def random_symbol():
    return random.choice(list(sf.get_semantic_robust_alphabet()))

def mutate_selfies(selfies_str):
    symbols = list(sf.split_selfies(selfies_str))
    op = random.choice(["insert", "delete", "replace"])
    if op == "insert":
        pos = random.randrange(len(symbols))
        symbols.insert(pos, random_symbol())
    elif op == "delete" and len(symbols) > 1:
        pos = random.randrange(len(symbols))
        symbols.pop(pos)
    elif op == "replace":
        pos = random.randrange(len(symbols))
        symbols[pos] = random_symbol()
    return ''.join(symbols)

def crossover_selfies(a, b):
    a_s = list(sf.split_selfies(a))
    b_s = list(sf.split_selfies(b))
    cut_a = random.randrange(len(a_s))
    cut_b = random.randrange(len(b_s))
    child = a_s[:cut_a] + b_s[cut_b:]
    return ''.join(child)