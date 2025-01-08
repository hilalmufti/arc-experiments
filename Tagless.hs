module Tagless where

class Symantics repr where
    int :: Int -> repr Int
    add :: repr Int -> repr Int -> repr Int
    lam :: (repr a -> repr b) -> repr (a -> b)
    app :: repr (a -> b) -> repr a -> repr b

newtype R a = R{unR :: a}

instance Symantics R where
    int x = R x
    add e1 e2 = R $ unR e1 + unR e2
    lam f = R $ unR . f . R
    app e1 e2 = R $ (unR e1) (unR e2)

newtype ShowExpr a = ShowExpr { showExpr :: String }

instance Symantics ShowExpr where
    int x = ShowExpr (show x)
    add e1 e2 = ShowExpr $ "(" ++ showExpr e1 ++ " + " ++ showExpr e2 ++ ")"
    lam f = ShowExpr "\\x -> ..."  -- simplification for lambda
    app e1 e2 = ShowExpr $ "(" ++ showExpr e1 ++ " " ++ showExpr e2 ++ ")"

eval :: R a -> a
eval e = unR e

-- Test expressions
test1 :: Symantics repr => repr Int
test1 = add (int 2) (int 3)

test2 :: Symantics repr => repr Int
test2 = add (add (int 1) (int 2)) (int 3)

main :: IO ()
main = do

    putStrLn "Testing R interpreter:"
    putStrLn $ "2 + 3 = " ++ show (eval test1)
    putStrLn $ "(1 + 2) + 3 = " ++ show (eval test2)
    
    putStrLn "\nTesting ShowExpr interpreter:"
    putStrLn $ "2 + 3 as expression: " ++ showExpr test1
    putStrLn $ "(1 + 2) + 3 as expression: " ++ showExpr test2