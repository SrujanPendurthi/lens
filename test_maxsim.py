import torch

from lens.rank import maxsim

q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
d0 = torch.tensor([[0.5, 0.5], [0.2, 0.1]])  # maxes: 0.5, 0.5 -> 1.0
d1 = torch.tensor([[0.9, 0.0], [0.0, 0.8]])  # maxes: 0.9, 0.8 -> 1.7 (planted gold)
s = maxsim(q, [d0, d1])
assert torch.allclose(s, torch.tensor([1.0, 1.7])), s
assert s.argmax() == 1
print("ok")
