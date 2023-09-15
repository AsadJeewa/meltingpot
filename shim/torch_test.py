import torch
from torch.distributions.categorical import Categorical

# define a torch tensor
t = torch.tensor([[[1],
   [2]],
   [[3],
   [4]]])
print("Tensor:", t)
print("Size of Tensor:", t.size())
#n steps x m agents x stacked x dim
#n steps x m agents x dim
# flatten the above tensor using start_dims
flatten_t = torch.flatten(t)
flatten_t0 = torch.flatten(t, start_dim=0, end_dim=1)
print("Flatten tensor:", flatten_t)
print(flatten_t.size())
print(flatten_t0)
print(flatten_t0.size())

print(t.size())
print(t[0].size())
print(t[0])

logits = torch.tensor([1,1,1]).float()
logits[0] = float('nan')#actor output is NAN!!!
if torch.isnan(logits).any():
   print("NAN")
   print(logits)
m = Categorical(logits=logits)#nan logits
print("probs =", m.probs)
print(m)