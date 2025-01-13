import os
import json
import inspect
from typing import Tuple, TypeVar, TypeAlias, Literal, Any

# TODO: these abstractions are not very reusable

EASY = {
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

GridM = list[list[int]]

# type Pair[T, U] = tuple[T, U]
Input = GridM
Output = GridM

# Sample = Pair[Input, Output]
SampleM = dict[str, GridM] # think of this as (inputs, outputs), but with named fields
type TaskM = list[SampleM] # list of input-output pairs

# TODO: make this more general
ARCTask = dict[str, TaskM] # task with train and test sets

TaskName = str
Dataset = dict[TaskName, ARCTask] # we'll have 400 train tasks, 400 test tasks


def is_easy(tn: TaskName) -> bool:
    return tn in EASY


def load_files(fs: list[str], loader=json.load) -> dict[str, Any]:
    return {f.split('/')[-1]: loader(open(f)) for f in fs}


def load_path(path: str, loader=json.load) -> dict[str, Any]:
    return load_files([f"{path}/{f}" for f in os.listdir(path)], loader)


def make_dataset(mode: Literal['training', 'evaluation'], path='data') -> Dataset:
    return {k[:-5]: v for k, v in load_path(f"{path}/{mode}").items()}


load_dataset = make_dataset


def get_easy(ds: Dataset) -> Dataset:
    return {k: v for k, v in ds.items() if is_easy(k)}


def make_easy_dataset() -> Dataset:
    return get_easy(make_dataset('training'))