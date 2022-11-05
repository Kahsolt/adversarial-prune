#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/27 

import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cuda':
  torch.backends.cudnn.enabled = True
  torch.backends.cudnn.benchmark = True


def float_to_str(x:str, n_prec:int=3) -> str:
  # integer
  if int(x) == x: return str(int(x))
  
  # float
  sci = f'{x:e}'
  frac, exp = sci.split('e')
  
  frac_r = round(float(frac), n_prec)
  frac_s = f'{frac_r}'
  if frac_s.endswith('.0'):   # remove tailing '.0'
    frac_s = frac_s[:-2]
  exp_i = int(exp)
  
  if exp_i != 0:
    # '3e-5', '-1.2e+3'
    return f'{frac_s}e{exp_i}'
  else:
    # '3.4', '-1.2'
    return f'{frac_s}'
