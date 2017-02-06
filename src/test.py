from itertools import chain
from timeit import timeit
import logging
import bisect
import numpy as np
from numpy.core.fromnumeric import sum
from numpy.core.umath import add

a = [1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
#a_2 = a[bisect.bisect_left(a, 3):bisect.bisect_right(a, 3)]

b = np.array(a)
print(b.ndim, b.shape, len(b.shape))


def trapz(y, x):
    """
    Integrate along the given axis using the composite trapezoidal rule.
    Integrate `y` (`x`) along given axis.
    Parameters
    ----------
    y : array_like
        Input array to integrate.
    x : array_like, optional
        If `x` is None, then spacing between all `y` elements is `dx`.
    Returns
    """
    y = np.asanyarray(y)
    x = np.asanyarray(x)
    d = np.diff(x)
    # reshape to correct shape
    shape = [1]
    shape[0] = d.shape[0]
    d = d.reshape(shape)
    slice1 = [slice(None)]
    slice2 = [slice(None)]
    slice1[0] = slice(1, None)
    slice2[0] = slice(None, -1)
    try:
        ret = (d * (y[slice1] + y[slice2]) / 2.0).sum(0)
    except ValueError:
        print("trapz value err")
        """
        # Operations didn't work, cast to ndarray
        d = np.asarray(d)
        y = np.asarray(y)
        ret = add.reduce(d * (y[slice1]+y[slice2])/2.0, axis)
        """
    return ret

a = np.negative(np.ones(5))
print(a)