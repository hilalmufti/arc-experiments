module Evaluator where

import Data.List (minimum, maximum)

-- Expression data type
data Expr = Add Expr Expr
          | Mul Expr Expr
          | Neg Expr
          | Sin Expr
          | Cos Expr
          | Const Double
          deriving (Show)

-- Evaluator type class
class Evaluator a where
    eval :: Expr -> a

-- Standard floating-point evaluation
instance Evaluator Double where
    eval (Const x) = x
    eval (Add e1 e2) = eval e1 + eval e2
    eval (Mul e1 e2) = eval e1 * eval e2
    eval (Neg e) = -(eval e)
    eval (Sin e) = sin (eval e)
    eval (Cos e) = cos (eval e)

-- Interval arithmetic
data Interval = Interval { lower :: Double, upper :: Double }
    deriving (Show)

instance Evaluator Interval where
    eval (Const x) = Interval x x
    eval (Add e1 e2) = let i1 = eval e1
                           i2 = eval e2
                       in Interval (lower i1 + lower i2) (upper i1 + upper i2)
    eval (Mul e1 e2) = let i1 = eval e1
                           i2 = eval e2
                           products = [lower i1 * lower i2, lower i1 * upper i2,
                                     upper i1 * lower i2, upper i1 * upper i2]
                       in Interval (minimum products) (maximum products)
    eval (Neg e) = let i = eval e
                   in Interval (-(upper i)) (-(lower i))
    eval (Sin e) = let i = eval e
                       vals = [sin (lower i), sin (upper i)]
                   in Interval (minimum vals) (maximum vals)
    eval (Cos e) = let i = eval e
                       vals = [cos (lower i), cos (upper i)]
                   in Interval (minimum vals) (maximum vals)

-- Symbolic differentiation
data SymbolicDiff = SymbolicDiff { expr :: Expr, deriv :: Expr }
    deriving (Show)

instance Evaluator SymbolicDiff where
    eval (Const x) = SymbolicDiff (Const x) (Const 0)
    eval (Add e1 e2) = let s1 = eval e1
                           s2 = eval e2
                       in SymbolicDiff (Add (expr s1) (expr s2))
                                     (Add (deriv s1) (deriv s2))
    eval (Mul e1 e2) = let s1 = eval e1
                           s2 = eval e2
                       in SymbolicDiff (Mul (expr s1) (expr s2))
                                     (Add (Mul (deriv s1) (expr s2))
                                         (Mul (expr s1) (deriv s2)))
    eval (Neg e) = let s = eval e
                   in SymbolicDiff (Neg (expr s)) (Neg (deriv s))
    eval (Sin e) = let s = eval e
                   in SymbolicDiff (Sin (expr s))
                                 (Mul (Cos (expr s)) (deriv s))
    eval (Cos e) = let s = eval e
                   in SymbolicDiff (Cos (expr s))
                                 (Neg (Mul (Sin (expr s)) (deriv s)))

-- Example usage
example :: Expr
example = Mul (Sin (Add (Const 1) (Const 2))) (Cos (Const 1))

-- Test function
test :: IO ()
test = do
    putStrLn "Standard evaluation:"
    print (eval example :: Double)
    
    putStrLn "\nInterval evaluation:"
    print (eval example :: Interval)
    
    putStrLn "\nSymbolic differentiation:"
    let diff = eval example :: SymbolicDiff
    putStrLn "Original:"
    print (expr diff)
    putStrLn "Derivative:"
    print (deriv diff)