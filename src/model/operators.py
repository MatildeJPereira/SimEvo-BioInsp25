# SELFIES-based variation operators: mutation and crossover.
# All operators return valid SELFIES strings by construction.

import random
import selfies as sf

def random_symbol():
    """
    Return a randomly chosen SELFIES token from a filtered alphabet.
    The alphabet is limited to common CHONPS atoms and ring/branch tokens
    to keep molecules chemically plausible while enabling diversity.
    """
    filtered_alphabet = {
        '[C]','[=C]','[#C]',        # neutral carbon variants
        '[N]','[=N]',               # nitrogen
        '[O]','[=O]',               # oxygen
        '[S]',                      # sulfur
        '[P]',                      # phosphorus
        '[Branch1]','[Branch2]',    # branching
        '[Ring1]','[Ring2]'}        # ring indicators
    return random.choice(list(filtered_alphabet))


def mutate_selfies(selfies_str):
    """
    Apply a random mutation: insert, delete, or replace a SELFIES symbol.
    Mutation always yields a syntactically valid SELFIES string.
    """
    symbols = list(sf.split_selfies(selfies_str))
    op = random.choice(["insert", "delete", "replace"])

    # INSERT: add a symbol at a random position
    if op == "insert":
        pos = random.randrange(len(symbols))
        symbols.insert(pos, random_symbol())

    # DELETE: remove a symbol if molecule would remain non-empty
    elif op == "delete" and len(symbols) > 1:
        pos = random.randrange(len(symbols))
        symbols.pop(pos)

    # REPLACE: overwrite an existing symbol with a new one
    elif op == "replace":
        pos = random.randrange(len(symbols))
        symbols[pos] = random_symbol()

    # Join into a new SELFIES string
    return ''.join(symbols)

def crossover_selfies(a, b):
    """
    Single‑point crossover between two SELFIES strings.

    Cuts each parent at an independent random point and concatenates
    prefix(a) with suffix(b). This operation always yields valid SELFIES.
    """
    a_s = list(sf.split_selfies(a))
    b_s = list(sf.split_selfies(b))
    cut_a = random.randrange(len(a_s))
    cut_b = random.randrange(len(b_s))
    child = a_s[:cut_a] + b_s[cut_b:]
    return ''.join(child)