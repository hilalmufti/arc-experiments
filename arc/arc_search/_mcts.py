import itertools
import logging
import math
import random
import time
import os
from collections import defaultdict
from datetime import datetime

import tqdm
import numpy as np

from arc.load_data import load_tasks
from arc.arc_search._grammar import read_functions
from arc.arc_dsl.constants import *
import arc.arc_dsl.dsl as dsl
from arc.arc_dsl.dsl import *
from arc.utils import set_seed

SEED = 546


class MCTSNode:
    def __init__(self, state=(), parent=None):
        self.state = state  # Tuple of primitive names applied so far
        self.parent = parent
        self.children = {}  # Maps actions (primitive names) to nodes
        self.visits = 0
        self.value = 0.0

    def __repr__(self):
        return f"MCTSNode({self.state}, visits={self.visits}, value={self.value}, children={set(self.children.keys())})"


def make_evaluator(task):
    ins = tuple(example["input"] for example in task["train"])
    outs = tuple(example["output"] for example in task["train"])

    def evaluator(program):
        return all(program(i) == o for i, o in zip(ins, outs))

    return evaluator


class MCTS:
    def __init__(
        self, primitive_names, max_depth, evaluation_fn, exploration_constant=1.41
    ):
        self.primitive_names = list(primitive_names)
        self.max_depth = max_depth
        self.evaluation_fn = evaluation_fn
        self.exploration_constant = exploration_constant
        self.Q = defaultdict(float)  # Total reward
        self.N = defaultdict(int)  # Visit count
        self.depth_primitives = defaultdict(lambda: defaultdict(int))

        self.setup_logging()
        self.stats = {
            "total_simulations": 0,
            "successful_simulations": 0,
            "nodes_created": 0,
            "max_depth_reached": 0,
            "unique_programs_tried": set(),
        }

    def setup_logging(self):
        if not os.path.exists("logs"):
            os.makedirs("logs")

        # Create a unique log file for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"logs/mcts_search_{timestamp}.log"

        # Setup logging configuration
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler(),  # Also print to console
            ],
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"Starting new MCTS search with {len(self.primitive_names)} primitives"
        )
        self.logger.info(
            f"Max depth: {self.max_depth}, Exploration constant: {self.exploration_constant}"
        )

    def log_statistics(self, iteration):
        stats = {
            "Iteration": iteration,
            "Success Rate": f"{(self.stats['successful_simulations'] / max(1, self.stats['total_simulations'])) * 100:.2%}",
            "Unique Programs": len(self.stats["unique_programs_tried"]),
            "Nodes Created": self.stats["nodes_created"],
            "Max Depth": self.stats["max_depth_reached"],
        }
        self.logger.info(f"Search statistics: {stats}")

    def log_final_statistics(self):
        self.logger.info("=== Final search statistics ===")
        self.logger.info(f"Total simulations: {self.stats['total_simulations']}")
        self.logger.info(
            f"Successful simulations: {self.stats['successful_simulations']}"
        )
        self.logger.info(
            f"Success rate: {(self.stats['successful_simulations'] / max(1, self.stats['total_simulations'])) * 100:.2%}"
        )
        self.logger.info(
            f"Unique programs tried: {len(self.stats['unique_programs_tried'])}"
        )
        self.logger.info(f"Total nodes created: {self.stats['nodes_created']}")
        self.logger.info(f"Max depth reached: {self.stats['max_depth_reached']}")

    def get_program_string(self, state):
        left_side = "".join([p + "(" for p in state])
        right_side = ")" * len(state)
        return f"lambda I: {left_side}I{right_side}"

    def try_program(self, state, error_value=0.0):
        program_string = self.get_program_string(state)
        self.stats["total_simulations"] += 1
        self.stats["unique_programs_tried"].add(program_string)
        self.stats["max_depth_reached"] = max(
            self.stats["max_depth_reached"], len(state)
        )
        try:
            program = eval(program_string)
            result = self.evaluation_fn(program)
            if result:
                self.stats["successful_simulations"] += 1
                self.logger.info(f"Found successful program: {program_string}")
            return float(result)
        except Exception as e:
            self.logger.debug(f"Program evaluation failed: {str(e)}")
            return error_value

    def get_ucb(self, node, child_action):
        if node.visits == 0 or self.N[(node.state, child_action)] == 0:
            return float("inf")
        exploitation = (
            self.Q[(node.state, child_action)] / self.N[(node.state, child_action)]
        )
        exploration = self.exploration_constant * math.sqrt(
            math.log(node.visits) / self.N[(node.state, child_action)]
        )
        return exploitation + exploration

    def select(self, node):
        path = []
        while len(node.state) < self.max_depth and len(node.children) == len(
            self.primitive_names
        ):
            # Select child with highest UCB value
            ucbs = {
                action: self.get_ucb(node, action) + random.uniform(0, 1e-6)
                for action in node.children
            }
            action = max(ucbs.keys(), key=lambda a: ucbs[a])
            child = node.children[action]
            path.append((node, self.get_ucb(node, action)))
            node = child

        if path:
            # print(path)
            self.logger.debug(
                f"Selection path: {' -> '.join([f'{a}(ucb={u:.2f})' for a, u in path])}"
            )
        return node

    def expand(self, node):
        if len(node.state) >= self.max_depth:
            return node

        self.logger.debug(
            f"Expanding node at depth {len(node.state)}: {self.get_program_string(node.state)}"
        )

        # Try each primitive as a possible action
        for primitive in self.primitive_names:
            if primitive not in node.children:
                new_state = node.state + (primitive,)
                new_node = MCTSNode(new_state, parent=node)
                node.children[primitive] = new_node
                self.stats["nodes_created"] += 1
                return new_node
        return node

    def simulate(self, node):
        current_state = list(node.state)
        simulation_path = []

        # Random rollout until max depth
        while len(current_state) < self.max_depth:
            action = random.choice(self.primitive_names)
            current_state.append(action)
            simulation_path.append(action)

            # print(current_state)

            res = self.try_program(current_state, error_value=-1.0)
            if self.try_program(current_state) == 1.0:
                self.logger.debug(f"Simulation path: {' -> '.join(simulation_path)}")
                return 1.0
            elif res == -1.0:
                return 0.0

        # Convert state to program string and evaluate
        return self.try_program(current_state, error_value=0.0)

    def backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            if node.parent is not None:
                action = node.state[-1]
                self.N[(node.parent.state, action)] += 1
                self.Q[(node.parent.state, action)] += reward
            node = node.parent

    def search(self, num_simulations):
        root = MCTSNode()
        programs_found = []

        self.logger.info(f"Starting search with {num_simulations} simulations")

        for i in range(num_simulations):
            if i % 1000 == 0:
                self.log_statistics(i)

            node = self.select(root)
            node = self.expand(node)
            reward = self.simulate(node)
            self.backpropagate(node, reward)

            # If we found a solution, add it to our list
            if reward == 1.0:
                program_string = self.get_program_string(node.state)
                programs_found.append(program_string)

        self.log_final_statistics()
        return programs_found

    def __repr__(self):
        return (
            f"MCTS({self.max_depth}, {self.exploration_constant}, {self.Q}, {self.N})"
        )


def i2p(index, itop):
    return itop[index]


def seq_to_program(primitives):
    program_strings = []
    for primitive in primitives:
        l = "".join([p + "(" for p in primitives])
        r = ")" * len(primitives)
        program_string = f"lambda I: {l}I{r}"
        program_strings.append(program_string)
    return program_strings


def solve_mcts(tasks, primitive_names, max_depth, num_simulations=1000):
    guesses = {}
    logger = logging.getLogger(__name__)

    tic = time.time()

    for key, task in tqdm.tqdm(tasks.items()):
        # for key, task in tasks.items():
        logger.info(f"\n=== Starting task {key} ===")
        logger.info(f"Training examples: {len(task['train'])}")

        mcts = MCTS(
            primitive_names,
            max_depth,
            make_evaluator(task),
            exploration_constant=math.sqrt(2),
        )
        solutions = mcts.search(num_simulations)

        if solutions:
            logger.info(f"Found {len(solutions)} candidate programs for task {key}")
            logger.info(f"First solution: {solutions[0]}")
            guesses[key] = min(solutions, key=len)
        else:
            logger.info(f"No solutions found for task {key}")

    toc = time.time()

    # print(f"\nMade guesses for {len(guesses)} tasks")
    logger.info(f"\nCompleted all tasks. Solved {len(guesses)}/{len(tasks)} tasks")
    logger.info(f"Total time taken: {toc - tic:.2f} seconds")
    return guesses


if __name__ == "__main__":
    set_seed(SEED)

    prims = read_functions(dsl)
    prim_names = {p.__name__ for p in prims}
    print(f"DSL consists of {len(prims)} primitives: {prim_names}")

    MAX_DEPTH = 20
    # program_strings = search(MAX_DEPTH, primitive_names)
    # print('\n'.join([*program_strings[:10], '...']))
    # print(f"Space to search consists of {len(program_strings)} programs:\n")
    # programs = {prog_str: eval(prog_str) for prog_str in program_strings}

    tasks = load_tasks("training")
    # guesses = solve(tasks, programs)
    guesses = solve_mcts(tasks, prim_names, MAX_DEPTH, num_simulations=4000000)
    print(guesses)
