#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/10/27 

import os
from argparse import ArgumentParser

import torch
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
from scipy.stats import normaltest

from model import MODELS, get_model
from util import float_to_str


@torch.inference_mode()
def prune(args):
  print(f'>> loading pretrained {args.model}')
  model = get_model(args.model).eval()

  def prune_layer(layer, name):
    nonlocal n_param, n_pruned

    x = getattr(layer, name, None)
    if x is None: return
    
    x_original = x

    def show(x_o, x_n):
      plt.subplot(121) ; plt.hist(x_o.flatten().numpy(), bins=50)
      plt.subplot(122) ; plt.hist(x_n.flatten().numpy(), bins=50)
      plt.show()

    # raw stats
    x_flat = x.flatten().numpy()
    avg, std = x_flat.mean(), x_flat.std()
    if args.show: print(f'avg: {avg}, std: {std}')
    
    # small values to zero
    mask1 = torch.zeros_like(x, dtype=torch.bool)
    if args.eps:
      mask1 = x.abs() < args.eps
      x = Parameter(x * ~mask1)

    # large values to upper limit
    mask2 = torch.zeros_like(x, dtype=torch.bool)
    if args.sigma:
      k2, p = normaltest(x_flat)
      if p < 1e-5:    # pass distribution check test if p is small enough
        lower = avg - args.sigma * std
        upper = avg + args.sigma * std
        mask2 = (x < lower) | (x > upper)
        x = Parameter(x.clamp(lower, upper))
    
    # random perturbate
    mask3 = torch.zeros_like(x, dtype=torch.bool)
    if args.rand:
      n = torch.empty_like(x).uniform_(-args.rand, args.rand)
      mask3 = n != 0.0
      x = Parameter(x + n)

    if args.show: show(x_original, x)

    n_param    += x.numel()
    n_modified = (mask1 | mask2 | mask3).sum().item()
    
    if n_modified:
      setattr(layer, name, x)
      n_pruned += n_modified

  n_param, n_pruned = 0, 0
  for layer in model.modules():
    prune_layer(layer, 'weight')
    prune_layer(layer, 'bias')

  print(f'>> prune ratio: {n_pruned / n_param:.4%}')

  if not args.show:
    suffix = ''
    suffix += f'_E{float_to_str(args.eps)}'   if args.eps   else ''
    suffix += f'_S{float_to_str(args.sigma)}' if args.sigma else ''
    suffix += f'_R{float_to_str(args.rand)}'  if args.rand else ''
    save_fp = os.path.join(args.out_path, f'{args.model}{suffix}.pth')
    print(f'>> saving to {save_fp}')
    torch.save(model.state_dict(), save_fp)


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-M', '--model', default='resnet18', choices=MODELS, help='model architecture')
  parser.add_argument('-E', '--eps',   default=0.0,  type=float, help='if |w| < eps, w = 0')
  parser.add_argument('-S', '--sigma', default=0.0,  type=float, help='w = w.clip(avg-sigma*std, avg+sigma*std) if w ~ N(mean, std)')
  parser.add_argument('-R', '--rand',  default=0.0,  type=float, help='w = w + U[-r, r]')
  parser.add_argument('--show', action='store_true', help='show figure only, does not save')
  parser.add_argument('--data_path', default='data')
  parser.add_argument('--out_path', default='out')
  args = parser.parse_args()

  os.makedirs(args.data_path, exist_ok=True)
  os.makedirs(args.out_path, exist_ok=True)

  params = [args.eps, args.sigma, args.rand]
  if sum(params) == 0.0:
    raise ValueError('at least one of must be set')
  if True in [x < 0.0 for x in params]:
    raise ValueError('params should be non-negative')

  prune(args)
