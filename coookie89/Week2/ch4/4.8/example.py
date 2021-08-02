# 4.8. Numerical Stability and Initialization

import torch
from d2l import torch as d2l

if __name__ == "__main__":

    print("Exploding Gradients------------------------------\n")
    M = torch.normal(0, 1, size=(4, 4))

    print("a single matrix ==>\n", M, "\n")
    for i in range(100):
        M = torch.mm(M, torch.normal(0, 1, size=(4, 4)))

    print("after multiplying 100 matrices ==>\n", M, "\n")

    print("Vanishing Gradients------------------------------\n")
    x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
    y = torch.sigmoid(x)
    y.backward(torch.ones_like(x))

    d2l.plot(
        x.detach().numpy(),
        [y.detach().numpy(), x.grad.numpy()],
        legend=["sigmoid", "gradient"],
        figsize=(4.5, 2.5),
    )
