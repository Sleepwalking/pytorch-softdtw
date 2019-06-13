import numpy as np
import torch
from numba import jit
from torch.autograd import Function

@jit(nopython = True)
def compute_softdtw(D, gamma):
  N = D.shape[0]
  M = D.shape[1]
  R = np.zeros((N + 2, M + 2)) + 1e8
  R[0, 0] = 0
  for j in range(1, M + 1):
    for i in range(1, N + 1):
      r0 = -R[i - 1, j - 1] / gamma
      r1 = -R[i - 1, j] / gamma
      r2 = -R[i, j - 1] / gamma
      rmax = max(max(r0, r1), r2)
      rsum = np.exp(r0 - rmax) + np.exp(r1 - rmax) + np.exp(r2 - rmax)
      softmin = - gamma * (np.log(rsum) + rmax)
      R[i, j] = D[i - 1, j - 1] + softmin
  return R

@jit(nopython = True)
def compute_softdtw_backward(D_, R, gamma):
  N = D_.shape[0]
  M = D_.shape[1]
  D = np.zeros((N + 2, M + 2))
  E = np.zeros((N + 2, M + 2))
  D[1:N + 1, 1:M + 1] = D_
  E[-1, -1] = 1
  R[:, -1] = -1e8
  R[-1, :] = -1e8
  R[-1, -1] = R[-2, -2]
  for j in range(M, 0, -1):
    for i in range(N, 0, -1):
      a0 = (R[i + 1, j] - R[i, j] - D[i + 1, j]) / gamma
      b0 = (R[i, j + 1] - R[i, j] - D[i, j + 1]) / gamma
      c0 = (R[i + 1, j + 1] - R[i, j] - D[i + 1, j + 1]) / gamma
      a = np.exp(a0)
      b = np.exp(b0)
      c = np.exp(c0)
      E[i, j] = E[i + 1, j] * a + E[i, j + 1] * b + E[i + 1, j + 1] * c
  return E[1:N + 1, 1:M + 1]

class _SoftDTW(Function):
  @staticmethod
  def forward(ctx, D, gamma = 1.0):
    dev = D.device
    dtype = D.dtype
    # gamma = torch.FloatTensor([gamma]).to(dev)
    gamma = torch.Tensor([gamma]).to(dev).type(dtype) # dtype fixed
    D_ = D.detach().cpu().numpy()
    g_ = gamma.item()
    # R = torch.FloatTensor(compute_softdtw(D_, g_)).to(dev)
    R = torch.Tensor(compute_softdtw(D_, g_)).to(dev).type(dtype)
    ctx.save_for_backward(D, R, gamma)
    return R[-2, -2]

  @staticmethod
  def backward(ctx, grad_output):
    dev = grad_output.device
    dtype = grad_output.dtype
    D, R, gamma = ctx.saved_tensors
    D_ = D.detach().cpu().numpy()
    R_ = R.detach().cpu().numpy()
    g_ = gamma.item()
    # E = torch.FloatTensor(compute_softdtw_backward(D_, R_, g_)).to(dev)
    E = torch.Tensor(compute_softdtw_backward(D_, R_, g_)).to(dev).type(dtype)
    return grad_output * E, None

## Added
def calc_distance_matrix(x, y):
  n = x.size(0)
  m = y.size(0)
  d = x.size(1)
  x = x.unsqueeze(1).expand(n, m, d)
  y = y.unsqueeze(0).expand(n, m, d)
  dist = torch.pow(x - y, 2).sum(2)
  return dist

'''
def calc_distance_matrices(xb, yb):
    batch_size = xb.size(0)
    n = xb.size(1)
    m = yb.size(1)
    D = torch.zeros(batch_size, n, m)
    for i in range(batch_size):
        D[i] = calc_distance_matrix(xb[i], yb[i])
    return D
'''

class SoftDTW(torch.nn.Module):
  def __init__(self, normalize=False):
    super(SoftDTW, self).__init__()
    self.normalize = normalize
    self.func_dtw = _SoftDTW.apply

  def forward(self, x, y):
    if self.normalize:
      D_xy = calc_distance_matrix(x, y)
      out_xy = self.func_dtw(D_xy)
      # normalize discrepancy to distance
      D_xx = calc_distance_matrix(x, x)
      out_xx = self.func_dtw(D_xx)
      D_yy = calc_distance_matrix(y, y)
      out_yy = self.func_dtw(D_yy)
      out = out_xy - 1/2 * (out_xx + out_yy)
      return out
    else:
      D_xy = calc_distance_matrix(x, y)
      out_xy = self.func_dtw(D_xy)
      return out_xy
