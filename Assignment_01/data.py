from os.path import join
import pandas as pd 

LS = "./ls_data/"
NLS = "./nls_data/"
REAL = "./real_world_data/"

def make_df(path, *classes, sep=","):
    dfs = []
    for c in classes:
        dfs.append(pd.read_csv(join(path, c), names=["x", "y"], sep=sep, dtype={"x" : "float", "y" : "float"}, engine="python"))
    return dfs

ls_data = make_df(LS, "class1.txt", "class2.txt")
nls_data = make_df(NLS, "class1.txt", "class2.txt")
real_data = make_df(REAL, "class1.txt", "class2.txt", "class3.txt", sep=r"\s")

