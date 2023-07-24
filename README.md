
# QuietAttention Experiments with nanoGPT

This repo is for my quick experiments with the ideas presented in [this blog post by Evan Miller](https://www.evanmiller.org/attention-is-off-by-one.html). It is a fork of [nanoGPT](https://github.com/karpathy/nanoGPT) with a small modification to add a QuietAttention implementation.

**NB: this is also something already in PyTorch! https://twitter.com/SamuelMullr/status/1683582347793530884?s=20 - https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html the PyTorch MHA has an `add_zero_attn` parameter. So this is not a new idea, although it was definitely new to me :)**

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

The very first test I ran seemed promising but on closer inspection I may have just been unlucky with the random samples I plotted, the notebooks now show the two approaches as about the same in terms of weight/activation distribution. Which means either I've done something silly (@Xenova already spotted one bug in my code) or there really isn't much difference being made. The latter sort of makes sense to me - this is the same as appending a single 0 to the logits, which in most cases I'd expect to have very little difference to the normal attention calculations. I'll try to train some larger models if I get a chance and look more closely at whether it actually helps with the thing it is supposed to affect: the outliers that hurt quantization accuracy. 

