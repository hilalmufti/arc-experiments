from enum import Enum, auto
from dataclasses import dataclass
import inspect

from nltk.grammar import CFG

# TODO: Add "O"
# TODO: fill in this cfg
    # [ ] literals

def count_params(fn):
    return len(inspect.signature(fn).parameters)


def show_funapp(ss):
    out = ""
    for name, n_params in ss.items():
        out += f'\'{name}\''
        if n_params > 0:
            out += ' \'(\''
            for _ in range(n_params - 1):
                out += ' Expr \', \''
            out += ' Expr \')\''
        out += ' | '
    return out[:-3]

def get_functions(module):
    return [
        obj
        for name, obj in inspect.getmembers(module)
        if inspect.isfunction(obj) and obj.__module__ == module.__name__
    ]

small_grammar = CFG.fromstring("""
Program -> OStmt Return | StmtList OStmt Return
StmtList -> Stmt | StmtList Stmt
Stmt -> Var ' = ' FunApp '; '
OStmt -> 'O' ' = ' FunApp '; '
Var -> 'x' Number
Number -> '1' | '2' | '3'
FunApp -> 'identity' '(' Expr ')' | 'add' '(' Expr ', ' Expr ')' | 'subtract' '(' Expr ', ' Expr ')' | Var '(' Expr ')'
Expr -> 'I' | Var | Lit
Lit -> 'F' | 'T' | 'ZERO' | 'ONE' | 'TWO' | 'THREE' | 'FOUR' | 'FIVE' | 'SIX' | 'SEVEN' | 'EIGHT' | 'NINE' | 'TEN' | 'NEG_ONE' | 'NEG_TWO'
Return -> 'return ' 'O'""")

# TODO: increase Number to 60 or more
# TODO: add back | Var '(' Expr ')' to FunApp
grammar = CFG.fromstring("""
Program -> OStmt Return | StmtList OStmt Return
StmtList -> Stmt | StmtList Stmt
Stmt -> Var ' = ' FunApp '; '
OStmt -> 'O' ' = ' FunApp '; '
Var -> 'x' Number
Number -> '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9' | '10'
FunApp -> 'add' '(' Expr ', ' Expr ')' | 'adjacent' '(' Expr ', ' Expr ')' | 'apply' '(' Expr ', ' Expr ')' | 'argmax' '(' Expr ', ' Expr ')' | 'argmin' '(' Expr ', ' Expr ')' | 'asindices' '(' Expr ')' | 'asobject' '(' Expr ')' | 'astuple' '(' Expr ', ' Expr ')' | 'backdrop' '(' Expr ')' | 'bordering' '(' Expr ', ' Expr ')' | 'both' '(' Expr ', ' Expr ')' | 'bottomhalf' '(' Expr ')' | 'box' '(' Expr ')' | 'branch' '(' Expr ', ' Expr ', ' Expr ')' | 'canvas' '(' Expr ', ' Expr ')' | 'cellwise' '(' Expr ', ' Expr ', ' Expr ')' | 'center' '(' Expr ')' | 'centerofmass' '(' Expr ')' | 'chain' '(' Expr ', ' Expr ', ' Expr ')' | 'cmirror' '(' Expr ')' | 'color' '(' Expr ')' | 'colorcount' '(' Expr ', ' Expr ')' | 'colorfilter' '(' Expr ', ' Expr ')' | 'combine' '(' Expr ', ' Expr ')' | 'compose' '(' Expr ', ' Expr ')' | 'compress' '(' Expr ')' | 'connect' '(' Expr ', ' Expr ')' | 'contained' '(' Expr ', ' Expr ')' | 'corners' '(' Expr ')' | 'cover' '(' Expr ', ' Expr ')' | 'crement' '(' Expr ')' | 'crop' '(' Expr ', ' Expr ', ' Expr ')' | 'decrement' '(' Expr ')' | 'dedupe' '(' Expr ')' | 'delta' '(' Expr ')' | 'difference' '(' Expr ', ' Expr ')' | 'divide' '(' Expr ', ' Expr ')' | 'dmirror' '(' Expr ')' | 'dneighbors' '(' Expr ')' | 'double' '(' Expr ')' | 'downscale' '(' Expr ', ' Expr ')' | 'either' '(' Expr ', ' Expr ')' | 'equality' '(' Expr ', ' Expr ')' | 'even' '(' Expr ')' | 'extract' '(' Expr ', ' Expr ')' | 'fgpartition' '(' Expr ')' | 'fill' '(' Expr ', ' Expr ', ' Expr ')' | 'first' '(' Expr ')' | 'flip' '(' Expr ')' | 'fork' '(' Expr ', ' Expr ', ' Expr ')' | 'frontiers' '(' Expr ')' | 'gravitate' '(' Expr ', ' Expr ')' | 'greater' '(' Expr ', ' Expr ')' | 'halve' '(' Expr ')' | 'hconcat' '(' Expr ', ' Expr ')' | 'height' '(' Expr ')' | 'hfrontier' '(' Expr ')' | 'hline' '(' Expr ')' | 'hmatching' '(' Expr ', ' Expr ')' | 'hmirror' '(' Expr ')' | 'hperiod' '(' Expr ')' | 'hsplit' '(' Expr ', ' Expr ')' | 'hupscale' '(' Expr ', ' Expr ')' | 'identity' '(' Expr ')' | 'inbox' '(' Expr ')' | 'increment' '(' Expr ')' | 'index' '(' Expr ', ' Expr ')' | 'ineighbors' '(' Expr ')' | 'initset' '(' Expr ')' | 'insert' '(' Expr ', ' Expr ')' | 'intersection' '(' Expr ', ' Expr ')' | 'interval' '(' Expr ', ' Expr ', ' Expr ')' | 'invert' '(' Expr ')' | 'last' '(' Expr ')' | 'lbind' '(' Expr ', ' Expr ')' | 'leastcolor' '(' Expr ')' | 'leastcommon' '(' Expr ')' | 'lefthalf' '(' Expr ')' | 'leftmost' '(' Expr ')' | 'llcorner' '(' Expr ')' | 'lowermost' '(' Expr ')' | 'lrcorner' '(' Expr ')' | 'manhattan' '(' Expr ', ' Expr ')' | 'mapply' '(' Expr ', ' Expr ')' | 'matcher' '(' Expr ', ' Expr ')' | 'maximum' '(' Expr ')' | 'merge' '(' Expr ')' | 'mfilter' '(' Expr ', ' Expr ')' | 'minimum' '(' Expr ')' | 'mostcolor' '(' Expr ')' | 'mostcommon' '(' Expr ')' | 'move' '(' Expr ', ' Expr ', ' Expr ')' | 'mpapply' '(' Expr ', ' Expr ', ' Expr ')' | 'multiply' '(' Expr ', ' Expr ')' | 'neighbors' '(' Expr ')' | 'normalize' '(' Expr ')' | 'numcolors' '(' Expr ')' | 'objects' '(' Expr ', ' Expr ', ' Expr ', ' Expr ')' | 'occurrences' '(' Expr ', ' Expr ')' | 'ofcolor' '(' Expr ', ' Expr ')' | 'order' '(' Expr ', ' Expr ')' | 'other' '(' Expr ', ' Expr ')' | 'outbox' '(' Expr ')' | 'paint' '(' Expr ', ' Expr ')' | 'pair' '(' Expr ', ' Expr ')' | 'palette' '(' Expr ')' | 'papply' '(' Expr ', ' Expr ', ' Expr ')' | 'partition' '(' Expr ')' | 'portrait' '(' Expr ')' | 'position' '(' Expr ', ' Expr ')' | 'positive' '(' Expr ')' | 'power' '(' Expr ', ' Expr ')' | 'prapply' '(' Expr ', ' Expr ', ' Expr ')' | 'product' '(' Expr ', ' Expr ')' | 'rapply' '(' Expr ', ' Expr ')' | 'rbind' '(' Expr ', ' Expr ')' | 'recolor' '(' Expr ', ' Expr ')' | 'remove' '(' Expr ', ' Expr ')' | 'repeat' '(' Expr ', ' Expr ')' | 'replace' '(' Expr ', ' Expr ', ' Expr ')' | 'righthalf' '(' Expr ')' | 'rightmost' '(' Expr ')' | 'rot180' '(' Expr ')' | 'rot270' '(' Expr ')' | 'rot90' '(' Expr ')' | 'sfilter' '(' Expr ', ' Expr ')' | 'shape' '(' Expr ')' | 'shift' '(' Expr ', ' Expr ')' | 'shoot' '(' Expr ', ' Expr ')' | 'sign' '(' Expr ')' | 'size' '(' Expr ')' | 'sizefilter' '(' Expr ', ' Expr ')' | 'square' '(' Expr ')' | 'subgrid' '(' Expr ', ' Expr ')' | 'subtract' '(' Expr ', ' Expr ')' | 'switch' '(' Expr ', ' Expr ', ' Expr ')' | 'toindices' '(' Expr ')' | 'toivec' '(' Expr ')' | 'tojvec' '(' Expr ')' | 'toobject' '(' Expr ', ' Expr ')' | 'tophalf' '(' Expr ')' | 'totuple' '(' Expr ')' | 'trim' '(' Expr ')' | 'ulcorner' '(' Expr ')' | 'underfill' '(' Expr ', ' Expr ', ' Expr ')' | 'underpaint' '(' Expr ', ' Expr ')' | 'uppermost' '(' Expr ')' | 'upscale' '(' Expr ', ' Expr ')' | 'urcorner' '(' Expr ')' | 'valmax' '(' Expr ', ' Expr ')' | 'valmin' '(' Expr ', ' Expr ')' | 'vconcat' '(' Expr ', ' Expr ')' | 'vfrontier' '(' Expr ')' | 'vline' '(' Expr ')' | 'vmatching' '(' Expr ', ' Expr ')' | 'vmirror' '(' Expr ')' | 'vperiod' '(' Expr ')' | 'vsplit' '(' Expr ', ' Expr ')' | 'vupscale' '(' Expr ', ' Expr ')' | 'width' '(' Expr ')' | Var '(' Expr ')'
Expr -> 'I' | Var | Lit
Lit -> 'F' | 'T' | 'ZERO' | 'ONE' | 'TWO' | 'THREE' | 'FOUR' | 'FIVE' | 'SIX' | 'SEVEN' | 'EIGHT' | 'NINE' | 'TEN' | 'NEG_ONE' | 'NEG_TWO' | 'DOWN' | 'RIGHT' | 'UP' | 'LEFT' | 'ORIGIN' | 'NEG_UNITY' | 'UP_RIGHT' | 'DOWN_LEFT' | 'ZERO_BY_TWO' | 'TWO_BY_ZERO' | 'TWO_BY_TWO' | 'THREE_BY_THREE'
Return -> 'return ' 'O'""")
                         
# [ ] constants
# [ ] functions + their signatures
# [ ] max program length
# [ ] whether or not you can call variables
def dsl_to_grammar(dsl):
    ...