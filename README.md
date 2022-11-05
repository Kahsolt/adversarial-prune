# adversarial-prune

    Could model weight prune technic defend against adversial attacks? 🤔

----

⚠ Probably NOT!

**Conclusions**:

- PGD主要攻击输入层和倒数第二层(Linear前一层)
  - 在输入层，对尽可能多的神经元造成微小扰动
  - 在倒数第二层(layer4)，产生异常大值从而改变Linear输出
- 网络层内部量化无法抵抗对抗攻击
  - 因为过度线性性无处不在……
- 尝试了一些模型权重修改无法抵抗对抗攻击
  - 稀疏化 (`E`类prune) 卵用没有，因为PGD只对大值权重感兴趣
  - 压抑大值权重(直接改weight)会使得精度严重下降 (`S`类prune)，说明训练时确实靠这些大值神经元独裁
  - 压抑大值输出(对特征图clamp)会使得精度些许下降，并在layer4严重改变特征图数值分布 (PGD想要产生大值，但因为数值上限被clamp界住了，就会使得整体数值均值上涨、即bn后的数值正态分布sigma变大)
- 

Naive & rude prune method:

```python
# suppress small values to zero (manually set `E`)
if |w| < E: w = 0.0

# suppress large values to upper limit (manually set `S`)
if w ~ N[μ, σ]: w = w.clamp(μ - S * σ, μ + S * σ)

# random perturbate (manually set `R`)
w = w + U[-R, R]
```

ResNet18:

| Model | prune ratio | Clean Accuracy | Remnant Accuracy | Prediction Change Rate | Strict Attack Success Rate (from correct to wrong) | Note |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| pretrained    | 0% | 91.80% | 0.20% | 99.78% | 94.80% |  |
| pretrained-qt | 0% | 91.20% | 0.20% | 99.78% | 95.20% | **transfer attack** |
| - | - | - | - | - | - | - |
| prune E=1e-5 |  0.0978% | 91.80% | 0.20% | 99.78% | 94.80% |  |
| prune E=1e-4 |  0.5217% | 91.80% | 0.20% | 99.78% | 94.80% |  |
| prune E=1e-3 |  4.7786% | 92.00% | 0.20% | 99.78% | 94.90% |  |
| prune E=5e-3 | 23.2845% | 92.00% | 0.20% | 99.78% | 95.50% |  |
| prune E=7e-3 | 31.9905% | 92.50% | 0.20% | 99.78% | 95.40% |  |
| prune E=1e-2 | 43.9910% | 90.70% | 0.20% | 99.78% | 94.70% | starts to hurts accuracy |
| prune E=2e-2 | 72.8967% | 37.10% | 0.10% | 99.73% | 72.70% |  |
| prune E=5e-2 | 95.9159% |  0.10% | 0.10% |  0.00% |  0.10% |  |
| - | - | - | - | - | - | - |
| prune S=9 | 0.0050% | 91.80% |  0.20% | 99.78% | 95.20% |  |
| prune S=7 | 0.0182% | 91.70% |  0.20% | 99.78% | 94.50% |  |
| prune S=5 | 0.0922% | 85.90% | 32.60% | 63.56% | 66.20% | large weight neurals vote out the results |
| prune S=3 | 0.9286% | 23.50% |  7.20% | 76.17% | 78.60% |  |
| - | - | - | - | - | - | - |
| prune R=1e-3 | 100.0% | 91.90% | 34.20% | 63.44% | 65.10% | NN is stable with perturbated weights |
| prune R=1e-2 | 100.0% | 87.10% | 30.60% | 66.13% | 68.70% |  |
| prune R=1e-1 | 100.0% |  0.20% |  0.10% | 50.00% | 45.00% |  |

----

by Armit
2022/10/27 
