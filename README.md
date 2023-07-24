
# QuietAttention Experiments with nanoGPT

This repo is for my quick experiments with the ideas presented in [this blog post by Evan Miller](https://www.evanmiller.org/attention-is-off-by-one.html). It is a fork of [nanoGPT](https://github.com/karpathy/nanoGPT) with a small modification to add my QuietAttention implementation.

Key code change:

```python

# The modified softmax version:
def surftmex(x, dim=-1):
  maxes = torch.max(x, dim, keepdim=True)[0]
  x_exp = torch.exp(x-maxes)
  x_exp_sum = torch.sum(x_exp, dim, keepdim=True)
  output_custom = x_exp/(torch.exp(-maxes)+x_exp_sum) # << The key bit is the +torch.exp(-maxes)
  return output_custom

# And then in the attention implementation we do:
if self.quiet:
    att = surftmex(att)
else:
    att = F.softmax(att, dim=-1)

```

You can play with this by running the notebook "QuietAttention Tests", easiest in colab here: https://colab.research.google.com/drive/1ArFoybiGCuNUgJZIbwJn5bXHDW3AQgdB?usp=sharing
