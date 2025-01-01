import tqdm

import itertools
import os
import time
import traceback
from itertools import product, islice
from multiprocessing import Pool


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
        for p, program_string in enumerate(programs):
            program = globals()[f'p{p}']
            try:
                if all(program(i) == o for i, o in zip(train_inputs, train_outputs)):
                    hypotheses.append(program_string)
            except Exception as e:
                n_error += 1
                pass
        # select first program for making predictions
        errors.append(n_error / len(programs))
        if len(hypotheses) > 0:
            print(f" found {len(hypotheses)} candidate programs for task {key}!")
            guesses[key] = hypotheses[0]
    print('Average error rate:', sum(errors) / len(errors))
    print(f"\nMade guesses for {len(guesses)} tasks")
    return guesses

def make_fnhead_str(name, *args):
    return f"def {name}({', '.join(args)})"

def fnbody_list_to_str(body):
    return "".join(body)

def make_fn_str(head, body):
    if isinstance(head, tuple):
        name, *args = head
        head = make_fnhead_str(name, *args)
    elif not isinstance(head, str):
        raise ValueError(f"Expected head to be a tuple or string, got {type(head)}")
    
    if isinstance(body, list):
        body = fnbody_list_to_str(body)
    elif not isinstance(body, str):
        raise ValueError(f"Expected body to be a list or string, got {type(body)}")
    
    out = f"{head}: {body}"
    return out

def make_programhead_str(i):
    return make_fnhead_str(f'p{i}', 'I')

def make_program_str(i, body):
    out = make_fn_str(make_programhead_str(i), fnbody_list_to_str(body))
    return out

def generate_chunk(args):
    start_idx, chunk_size, grammar = args
    return [
        f"def p{i}(I): {''.join(p)}"
        for i, p in enumerate(
            islice(generate(grammar), chunk_size),
            start=start_idx
        )
    ]

def generate_program_strings_parallel(grammar, n_programs, chunk_size=10000, num_processes=None):
    num_chunks = (n_programs + chunk_size - 1) // chunk_size
    chunks = [
        (i * chunk_size, 
         min(chunk_size, n_programs - i * chunk_size),
         grammar)
        for i in range(num_chunks)
    ]
    
    with Pool(processes=num_processes) as pool:
        results = list(
            tqdm.tqdm(
                pool.imap(generate_chunk, chunks),
                total=num_chunks,
                ncols=80
            )
        )
    
    # Flatten results
    program_strings = [
        prog_str 
        for chunk in results 
        for prog_str in chunk
    ]
    
    return program_strings

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

    # n_programs = (2**27 + 2**28) // 2
    n_programs = 2**17

    with timer(f"Generated {n_programs} programs in:"):
        generated = tqdm.tqdm(enumerate(generate(grammar, n=n_programs)), total=n_programs, ncols=100, desc="Generating programs")
        program_strings = [make_program_str(i, p) for i, p in generated]

    with timer(f"Evaluated {n_programs} programs in:"):
        for ps in tqdm.tqdm(program_strings, total=n_programs, ncols=100, desc="Evaluating programs"):
            exec(ps)

    print(f"Space to search consists of {len(program_strings)} programs:\n")
    print('\n'.join([*program_strings[:5], '...']))
    print('\n'.join([*program_strings[-5:]]))
    print()

    n_procs = os.cpu_count()

    # programs = {prog_str: exec(prog_str) for prog_str in tqdm.tqdm(program_strings, total=len(program_strings), ncols=80)}

    tasks = load_tasks("training")
    # guesses = solve(tasks, programs)
    guesses = solve(tasks, program_strings)
    print(guesses)
