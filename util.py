from typing import Callable, NamedTuple, Union, Iterable, Mapping, Any


import mlx.core as mx
import mlx.utils as mx_util

import mlx.core.random as random

Array = mx.array
ArrayLike = mx.array | float 

ArrayTree = Array | Iterable["ArrayTree"] | Mapping[Any, "ArrayTree"]
ArrayLikeTree = ArrayLike | Iterable["ArrayLikeTree"] | Mapping[Any, "ArrayLikeTree"]


def norm_logpdf(x: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array:
    """Compute logpdf of multivariate distribution with means `loc` and variances `scale`. 
    
    NOTE: Right now we do not use any broadcasting rules and assert that x, loc, and 
    scale are all the same length."""
    ndim_in_range = lambda t: 0 < t.ndim <= 2
    assert ndim_in_range(x); assert ndim_in_range(loc); assert ndim_in_range(scale)
    assert len(x) == len(loc) == len(scale)

    # Functionally important part...
    # Follows jax.scipy.stats implementation 
    # Link: https://jax.readthedocs.io/en/latest/_modules/jax/_src/scipy/stats/norm.html#logpdf
    scale_sqrd = mx.square(scale)
    log_normalizer = mx.log(2 * mx.pi * scale_sqrd)
    quadratic = mx.square(x-loc) / scale_sqrd
    return (log_normalizer + quadratic) /  -2


# TODO: still need to verify this works as intended.
def normal_like(tree: ArrayLikeTree) -> ArrayTree:
    _normal_like = lambda x: random.normal(x.shape)
    return mx_util.tree_map(_normal_like, tree)
