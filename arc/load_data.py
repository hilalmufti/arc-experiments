import os
import json
import inspect

easy = {
    "67a3c6ac",
    "68b16354",
    "74dd1130",
    "3c9b0459",
    "6150a2bd",
    "9172f3a0",
    "9dfd6313",
    "a416b8f3",
    "b1948b0a",
    "c59eb873",
    "c8f0f002",
    "d10ecb37",
    "d511f180",
    "ed36ccf7",
    "4c4377d9",
    "6d0aefbc",
    "6fa7a44f",
    "5614dbcf",
    "5bd6f4ac",
    "5582e5ca",
    "8be77c9e",
    "c9e6f938",
    "2dee498d",
    "1cf80156",
    "32597951",
    "25ff71a9",
    "0b148d64",
    "1f85a75f",
    "23b5c85d",
    "9ecd008a",
    "ac0a08a4",
    "be94b721",
    "c909285e",
    "f25ffba3",
    "c1d99e64",
    "b91ae062",
    "3aa6fb7a",
    "7b7f7511",
    "4258a5f9",
    "2dc579da",
    "28bf18c6",
    "3af2c5a8",
    "44f52bb0",
    "62c24649",
    "67e8384a",
    "7468f01a",
    "662c240a",
    "42a50994",
    "56ff96f3",
    "50cb2852",
    "4347f46a",
    "46f33fce",
    "a740d043",
    "a79310a0",
    "aabf363d",
    "ae4f1146",
    "b27ca6d3",
    "ce22a75a",
    "dc1df850",
    "f25fbde4",
    "44d8ac46",
    "1e0a9b12",
    "0d3d703e",
    "3618c87e",
    "1c786137",
}

def get_taskname(filename: str) -> str:
    return filename[:-5]

def load_tasks(mode):
    tasks = {}
    prefix = "../" if not os.path.exists("data") else ""
    if mode == "easy":
        is_easy = True
        mode = "training"
    else:
        is_easy = False
    for f in os.listdir(prefix + f"data/{mode}"):
        if is_easy and get_taskname(f) not in easy:
            continue
        with open(prefix + f"data/{mode}/{f}") as fp:
            task = json.load(fp)
            task["train"] = tuple(task["train"])
            task["test"] = tuple(task["test"])
            for dct in task["train"]:
                dct["input"] = tuple(map(tuple, dct["input"]))
                dct["output"] = tuple(map(tuple, dct["output"]))
            for dct in task["test"]:
                dct["input"] = tuple(map(tuple, dct["input"]))
                dct["output"] = tuple(map(tuple, dct["output"]))
            tasks[get_taskname(f)] = task
    return tasks


def load_solutions(mode):
    directory = ("../" if not os.path.exists("data") else "") + f"data/{mode}"

    solutions = {}
    for f in os.listdir(directory):
        with open(f"{directory}/{f}") as fp:
            solution = json.load(fp)
            solutions[get_taskname(f)] = solution
    return solutions
