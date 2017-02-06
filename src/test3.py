import numpy as np


def compare_intersect(x, y):
    return frozenset(x).intersection(y)

a = [1,2,2,3,4]
b = [1,1,2,2,3]

print(compare_intersect(a,b))