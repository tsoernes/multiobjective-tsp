from itertools import chain
from timeit import timeit
import logging
import bisect
import numpy as np
from matplotlib import pyplot as plt
import itertools
from collections import defaultdict

a = [1,2,3,4,5]
b = itertools.tee(a)
for x, y in zip(itertools.islice(a, 0, None, 2), itertools.islice(a, 1, None, 2)):
    print(x, y)