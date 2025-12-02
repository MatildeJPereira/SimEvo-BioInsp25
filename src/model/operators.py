# Mutation & crossover on SELFIES: 
# - mutation: substitute, insert, delete 
# crossover: cut SELFIES at random position
# occasional sanity check by converting back to RDKit

import random
import selfies as sf

def random_symbol():
    filtered_alphabet = {'[C]','[=C]','[#C]',   # neutral C
                         '[N]','[=N]',          # neutral N
                         '[O]','[=O]',          # neutral O
                         '[S]','[=S]',          # neutral S
                         # optionally, just a few ions:
                         # '[N+1]', '[O-1]',
                          '[Branch1]','[Branch2]',
                          '[Ring1]','[Ring2]'}
# previous alphabet
#    {'[#Branch1]','[#Branch2]','[#Branch3]','[#C+1]','[#C-1]','[#C]','[#N+1]','[#N]','[#O+1]',
#                         '[#P+1]','[#P-1]','[#P]','[#S+1]','[#S-1]','[#S]','[=Branch1]','[=Branch2]','[=Branch3]',
#                         '[=C+1]','[=C-1]','[=C]','[=N+1]','[=N-1]','[=N]','[=O+1]','[=O]','[=P+1]','[=P-1]','[=P]',
#                         '[=Ring1]','[=Ring2]','[=Ring3]','[=S+1]','[=S-1]','[=S]','[Branch1]','[Branch2]','[Branch3]',
#                         '[C+1]','[C-1]','[C]','[N+1]','[N-1]','[N]','[O+1]','[O-1]','[O]','[P+1]','[P-1]','[P]',
#                         '[Ring1]','[Ring2]','[Ring3]','[S+1]','[S-1]','[S]'}
    
    return random.choice(list(filtered_alphabet))


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