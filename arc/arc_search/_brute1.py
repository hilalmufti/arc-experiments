import tqdm

import itertools
import os
import time
import traceback
from itertools import product, islice


from nltk.grammar import CFG
from nltk.parse.generate import generate
from tqdm.contrib.concurrent import process_map

from arc.load_data import load_tasks
from arc.arc_search._grammar import read_functions, read_constants, count_params, dsl2cfg
from arc.utils import set_seed, timer

from arc.arc_dsl.dsl import *
from arc.arc_dsl.constants import *
import arc.arc_dsl.dsl as dsl
import arc.arc_dsl.constants as constants

SEED = 546

def search(depth, primitive_names):
    program_strings = []
    for depth in tqdm.trange(1, depth + 1):
        primitive_tuples = product(*[primitive_names] * depth)
        for primitives in primitive_tuples:
            left_side = "".join([p + "(" for p in primitives])
            right_side = ")" * depth
            program_string = f"lambda I: {left_side}I{right_side}"
            program_strings.append(program_string)
    return program_strings

def solve(tasks, programs):
    guesses = dict()
    errors = []
    for key, task in (pbar := tqdm.tqdm(tasks.items(), ncols=80)):
        pbar.set_description(f"Solving task {key}")
        train_inputs = tuple((example["input"] for example in task["train"]))
        train_outputs = tuple((example["output"] for example in task["train"]))
        hypotheses = []
        # iterate over all programs
        # for program_string, program in tqdm.tqdm(programs.items()):
        n_error = 0
        for p, program_string in tqdm.tqdm(enumerate(programs), total=len(programs), ncols=80, desc="Solving..."):
            program = globals()[f'p{p}']
            try:
                if all(program(i) == o for i, o in zip(train_inputs, train_outputs)):
                    hypotheses.append(program_string)
            except Exception as e:
                n_error += 1
                pass
        # select first program for making predictions
        if len(hypotheses) > 0:
            print(f" found {len(hypotheses)} candidate programs for task {key}!")
            guesses[key] = hypotheses[0]
        errors.append(n_error / len(programs))
    print('Average error rate:', sum(errors) / len(errors))
    print(f"\nMade guesses for {len(guesses)} tasks")
    return guesses

def make_fnstr_head(name, *args):
    return f"def {name}({', '.join(args)})"

def fnstr_join_body(body):
    return "".join(body)

def make_fnstr(head, body):
    if isinstance(head, tuple):
        name, *args = head
        head = make_fnstr_head(name, *args)
    elif not isinstance(head, str):
        raise TypeError(f"Expected head to be a tuple or string, got {type(head)}")
    
    if isinstance(body, list):
        body = fnstr_join_body(body)
    elif not isinstance(body, str):
        raise TypeError(f"Expected body to be a list or string, got {type(body)}")
    
    out = f"{head}: {body}"
    return out

def make_progstr_head(i):
    return make_fnstr_head(f'p{i}', 'I')

def make_progstr(i, body):
    out = make_fnstr(make_progstr_head(i), fnstr_join_body(body))
    return out

if __name__ == "__main__":
    set_seed(SEED)

    prims = read_functions(dsl)
    prim_names = {p.__name__ for p in prims}
    print(f"DSL consists of {len(prims)} primitives: {sorted(prim_names)}")

    MAX_DEPTH = 3
    # program_strings = search(MAX_DEPTH, prim_names)

    fs = {f.__name__: count_params(f) for f in read_functions(dsl)}
    cs = read_constants(constants)
    prog_len = 2

    cfg_str = dsl2cfg(fs, cs, prog_len)
    grammar = CFG.fromstring(cfg_str)

    # n_programs = (2**27 + ((2**27 + 2**28) // 2)) // 2
    n_programs = 2**26
    # n_programs = 2**17

    with timer(f"Generated {n_programs} programs in:"):
        generated = enumerate(generate(grammar, n=n_programs))
        program_strings = [make_progstr(i, p) for i, p in tqdm.tqdm(generated, total=n_programs, ncols=100, desc="Generating programs")]

    with timer(f"Evaluated {n_programs} programs in:"):
        for ps in tqdm.tqdm(program_strings, total=n_programs, ncols=100, desc="Evaluating programs"):
            exec(ps)

    print(f"|search space| = {len(program_strings)} programs:\n")
    print('\n'.join([*program_strings[:5], '...']))
    print('\n'.join([*program_strings[-5:]]))
    print()

    tasks = load_tasks("training")
    with timer(f"Solved {len(tasks)} tasks in:"):
        guesses = solve(tasks, program_strings)
    print(guesses)
