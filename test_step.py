#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/27 

from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import normaltest

from model import get_model, MODELS
from data import get_dataloader, normalize, imshow
from util import device

from test import pgd

cpu = 'cpu'   # for qt model

def bn_forward(bn, x, name):
  avg = bn.running_mean[None, :, None, None]
  std = (bn.running_var[None, :, None, None] + 1e-8) ** 0.5
  w   = bn.weight      [None, :, None, None]
  b   = bn.bias        [None, :, None, None]

  #x_n = (x - avg) / std
  x_n = x
  for i in range(len(bn.bias)):
    x_n_np = x_n[i].detach().flatten().cpu().numpy()
    _, p = normaltest(x_n_np)
    print(f'{name} p={p}')

  return ((x - avg) / std) * w + b

def round_forward(model, x, ax, n_prec=1, show=False):
  self = model

  def draw(x, ax, title=''):
    if not show: return

    dx = (x - ax).abs()
    print(f'|fm_X - fm_AX|: max: {dx.max()}, mean: {dx.mean()}')

    x_np  = x .flatten().detach().cpu().numpy()
    ax_np = ax.flatten().detach().cpu().numpy()
    plt.subplot(121) ; plt.hist(x_np,  bins=32)
    plt.subplot(122) ; plt.hist(ax_np, bins=32)
    plt.suptitle(title)
    plt.show()
  
  #torch.round_(x,  decimals=3)
  #torch.round_(ax, decimals=3)

  ax = self.conv1(ax)   #; torch.round_(ax, decimals=n_prec)
  x  = self.conv1(x)    #; torch.round_(x, decimals=n_prec)
  draw(x, ax, 'conv1')

  for i in range(x.shape[1]):
    x_  = x[:, i:i+1, :, :]
    ax_ = ax[:, i:i+1, :, :]
    dx = (x_ - ax_).abs()
    dx_abs = dx.abs()
    
    print(f'channel-{i}')
    print('   |x - ax|:', dx_abs.min().item(),  dx_abs.max().item(),  dx_abs.mean().item(),  dx.std().item())
    print('   x:',   x_.min().item(),  x_.max().item(),  x_.mean().item(),  x_.std().item())
    print('   ax:', ax_.min().item(), ax_.max().item(), ax_.mean().item(), ax_.std().item())

    x_  = ( x_ -  x_.min()) / ( x_.max() -  x_.min())
    ax_ = (ax_ - ax_.min()) / (ax_.max() - ax_.min())
    
    imshow(x_, ax_, f'conv1 channel-{i}')

  ax = self.bn1(ax)     ; torch.round_(ax, decimals=n_prec)
  x = self.bn1(x)     ; torch.round_(x, decimals=n_prec) ; draw(x, ax, 'bn1')
  ax = self.relu(ax)    ; torch.round_(ax, decimals=n_prec)
  x = self.relu(x)    ; torch.round_(x, decimals=n_prec) ; draw(x, ax, 'relu')
  ax = self.maxpool(ax) ; torch.round_(ax, decimals=n_prec)
  x = self.maxpool(x) ; torch.round_(x, decimals=n_prec) ; draw(x, ax, 'maxpool')
  ax = self.layer1(ax)  ; torch.round_(ax, decimals=n_prec)

  x = self.layer1(x)  ; torch.round_(x, decimals=n_prec) ; draw(x, ax, 'layer1')
  ax = self.layer2(ax)  ; torch.round_(ax, decimals=n_prec)
  x = self.layer2(x)  ; torch.round_(x, decimals=n_prec) ; draw(x, ax, 'layer2')
  ax = self.layer3(ax)  ; torch.round_(ax, decimals=n_prec)
  x = self.layer3(x)  ; torch.round_(x, decimals=n_prec) ; draw(x, ax, 'layer3')
  ax = self.layer4(ax)  ; torch.round_(ax, decimals=n_prec)
  x = self.layer4(x)  ; torch.round_(x, decimals=n_prec) ; draw(x, ax, 'layer4')

  ax = self.avgpool(ax) ; torch.round_(ax, decimals=n_prec)
  x = self.avgpool(x) ; torch.round_(x, decimals=n_prec) ; draw(x, ax, 'avgpool')
  ax = ax.flatten(1)    ; torch.round_(ax, decimals=n_prec)
  x = x.flatten(1)    ; torch.round_(x, decimals=n_prec)
  ax = self.fc(ax)      ; torch.round_(ax, decimals=n_prec)
  x = self.fc(x)      ; torch.round_(x, decimals=n_prec) ; draw(x, ax, 'fc')

  return x, ax


def do_test(model, dataloader, model_qt=None, show=False) -> tuple:
  ''' Clean Accuracy, Remnant Accuracy, Attack Success Rate, Prediction Change Rate '''

  total, correct, rcorrect, changed = 0, 0, 0, 0
  attacked = 0

  model.eval()
  for X, Y in tqdm(dataloader):
    X = X.to(device)
    Y = Y.to(device)

    AX = pgd(model, X, Y, args.eps, args.alpha, args.steps)
    if show:
      with torch.no_grad():
        dx = AX - X
        Linf = dx.abs().max(dim=0)[0].mean()
        L2   = dx.square().sum(dim=0).sqrt().mean()
        print(f'Linf: {Linf}')
        print(f'L2: {L2}')

      imshow(X, AX)

    with torch.inference_mode():
      if model_qt:
        pred    = model_qt(normalize(X) .to(cpu)).argmax(dim=-1).to(device)
        pred_AX = model_qt(normalize(AX).to(cpu)).argmax(dim=-1).to(device)
      else:
        #forward = model
        forward = lambda x, ax: round_forward(model, x, ax, n_prec=args.n_prec, show=args.show)
  
        preds = forward(normalize(X), normalize(AX))
        pred    = preds[0].argmax(dim=-1)
        pred_AX = preds[1].argmax(dim=-1)

    total    += len(pred)
    correct  += (pred    == Y   ).sum().item()               # clean correct
    rcorrect += (pred_AX == Y   ).sum().item()               # adversarial still correct
    changed  += (pred_AX != pred).sum().item()               # prediction changed under attack
    attacked += ((pred == Y) * (pred_AX != Y)).sum().item()  # clean correct but adversarial wrong

    if show:
      print('Y:', Y)
      print('pred:', pred)
      print('pred_AX:', pred_AX)
      print(f'total: {total}, correct: {correct}, rcorrect: {rcorrect}, changed: {changed}, attacked: {attacked}')

  return [
    correct  / total   if total else 0,
    rcorrect / total   if total else 0,
    changed  / total   if total else 0,
    attacked / correct if correct else 0,
  ]


def test(args):
  ''' Model '''
  model    = get_model(args.model         ).to(device)
  model_qt = get_model(args.model, qt=True).to(cpu) if args.qt else None

  ''' Model '''
  if args.ckpt:
    print(f'>> loading chpt from {args.ckpt}')
    ckpt = torch.load(args.ckpt)
    model.load_state_dict(ckpt)

  ''' Data '''
  dataloader = get_dataloader(args.data_path, args.batch_size, shuffle=args.shuffle)
  
  ''' Test '''
  acc, racc, asr, pcr = do_test(model, dataloader, model_qt=model_qt, show=args.show)
  print(f'Clean Accuracy:         {acc:.2%}')
  print(f'Remnant Accuracy:       {racc:.2%}')
  print(f'Prediction Change Rate: {pcr:.2%}')
  print(f'Attack Success Rate:    {asr:.2%}')


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-M', '--model', default='resnet18', choices=MODELS, help='model to attack')
  parser.add_argument('--qt', action='store_true', help='use the qt version model')
  parser.add_argument('--ckpt', default=None, help='path to ckpt file')
  
  parser.add_argument('--eps',    type=float, default=0.03)
  parser.add_argument('--alpha',  type=float, default=0.001)
  parser.add_argument('--steps',  type=int,   default=10)
  parser.add_argument('--n_prec', type=int,   default=3, help='0 does not work, 1 is just ok')
  parser.add_argument('--show',   action='store_true')
  
  parser.add_argument('-B', '--batch_size', type=int, default=64)
  parser.add_argument('--shuffle', action='store_true')
  parser.add_argument('--data_path', default='data', help='folder path to downloaded dataset')
  parser.add_argument('--log_path', default='log', help='folder path to local trained model weights and logs')
  args = parser.parse_args()

  print('[Ckpt] use pretrained weights from torchvision/torchhub')

  test(args)
