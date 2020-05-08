pytorch-softdtw
===

An implementation of SoftDTW [1] for PyTorch. Should run pretty fast.

More goodies: check out [pytorch-softdtw-cuda](https://github.com/Maghoumi/pytorch-softdtw-cuda) by Maghoumi, a heavily upgraded version that runs parallel on CUDA.

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
from soft_dtw import SoftDTW
...
criterion = SoftDTW(gamma=1.0, normalize=True) # just like nn.MSELoss()
...
loss = criterion(out, target)
```


### Does it support pruning?

No. You can mess with the loop (line 13 and line 35) yourself.

License
---

Look, I just took their paper and wrote a single-file Python thingy. If you want to say thanks, which is of course welcome, then buy me a drink. Not a big fan of beer though. Umeshu in soda will be nice. Yeah sweety sweety.

Reference
---

[1] M. Cuturi and M. Blondel. "Soft-DTW: a Differentiable Loss Function for Time-Series". In Proceedings of ICML 2017.

