"""Misc. numeric functions."""
import numpy as np


def _reshape1d(v, n):
    """Reshape input v in size (n,) if possible."""
    try:
        l = len(v)
        if l == 1:
            w = v * np.ones(n, dtype=v.dtype)
        else:
            raise ValueError('Uncompatible size')
    except AttributeError:
        w = v * np.ones(n, dtype=type(v))
    return w


def reshape(v, nr=None, nc=None):
    """Reshape input v in size (nr, nc) if possible."""
    if nr is None and nc is None:
        raise ValueError()
    if nr is None:
        w = _reshape1d(v, nc)
    elif nc is None:
        w = _reshape1d(v, nr)
    else:
        try:
            s = v.shape
            if len(s) == 1:
                if nr == s[0]:
                    w = np.column_stack(nc * (v,))
                elif nc == s[0]:
                    w = np.row_stack(nr * (v,))
            else:
                w = np.reshape(v, (nr, nc))
        except AttributeError:
            w = v * np.ones((nr, nc), dtype=type(v))
    return w
