from enum import Enum, auto
from dataclasses import dataclass

from nltk.grammar import CFG

# TODO: Add "O"
# TODO: fill in this cfg
    # [ ] literals

grammar = CFG.fromstring("""
Program -> StmtList OStmt Return | OStmt Return
StmtList -> Stmt | StmtList Stmt
Stmt -> Var '=' FunApp
OStmt -> 'O' '=' FunApp
Var -> 'x' Number
Number -> '1' | '2' | '3'
FunApp -> 'identity' '(' Expr ')' | 'add' '(' Expr ',' Expr ')' | 'subtract' '(' Expr ',' Expr ')' | Var '(' Expr ')'
Expr -> 'I' | Var | Lit
Lit -> 'F' | 'T' | 'ZERO' | 'ONE' | 'TWO' | 'THREE' | 'FOUR' | 'FIVE' | 'SIX' | 'SEVEN' | 'EIGHT' | 'NINE' | 'TEN' | 'NEG_ONE' | 'NEG_TWO'
Return -> 'return' 'O'""")
                         
def dsl_to_grammar(dsl):
    ...