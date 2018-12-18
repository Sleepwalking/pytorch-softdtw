pytorch-softdtw
===

An implementation of SoftDTW [1] for PyTorch. Should run pretty fast.

Install
---

Just paste this file around. I don't believe in Python's package managers. Setting up the environment took me longer than actually writing this thing.

Depends on PyTorch and Numba.

How to use
---

`SoftDTW` autograd function computes the smoothed DTW distance (scalar) for a given distance matrix and calling backward on the result gives you the derivative (matrix) of the DTW distance (scalar) with respect to the distance matrix.

As the original authors pointed out [1], the derivative is the same as the expected DTW path. This is comparable to forward-backward algorithm for HMM.

You may also specify the temperature gamma (positive number). As gamma goes to zero, the result converges to that of the original "hard" DTW.

```python
from softdtw import SoftDTW

# compute D using whatever metric (L1, L2, ...)

func_dtw = SoftDTW.apply

R = func_dtw(D, 0.1) # the second argument is gamma
R.backward()

print(D.grad)
```

### Does it support pruning?

No. You can mess with the loop (line 13 and line 35) yourself.

License
---

Look, I just took their paper and wrote a single-file Python thingy. If you want to say thanks, which is of course welcome, then buy me a drink. Not a big fan of beer though. Umeshu in soda will be nice. Yeah sweety sweety.

Reference
---

[1] M. Cuturi and M. Blondel. "Soft-DTW: a Differentiable Loss Function for Time-Series". In Proceedings of ICML 2017.

