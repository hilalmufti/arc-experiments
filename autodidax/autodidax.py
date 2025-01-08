from collections.abc import Sequence
from contextlib import contextmanager
from typing import Any, NamedTuple

class Evaluator(NamedTuple):
    level: int
    trace_type: type['Trace']
    global_data: Any | None



class Binding:
    ...

c

trace_stack: list[Evaluator] = []

class Primitive(NamedTuple):
    name: str

add_p = Primitive('add')
mul_p = Primitive('mul')
neg_p = Primitive('neg')
sin_p = Primitive('sin')
cos_p = Primitive('cos')
reduce_sum_p = Primitive('reduce_sum')
greater_p = Primitive('greater')
less_p = Primitive('less')
transpose_p = Primitive('transpose')
broadcast_p = Primitive('broadcast')

def apply1(prim, *a, **kw):
    out, = apply(prim, *a, **kw)
    return out