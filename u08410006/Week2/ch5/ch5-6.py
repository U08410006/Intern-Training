import torch
from torch import nn


def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f"cuda:{i}")
    return torch.device("cpu")


def try_all_gpus():
    """Return all available GPUs, or [cpu(),] if no GPU exists."""
    devices = [
        torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())
    ]
    return devices if devices else [torch.device("cpu")]


if __name__ == "__main__":
    print(torch.device("cpu"), torch.device("cuda"), torch.device("cuda:1"))
    print(torch.cuda.device_count())
    try_gpu(), try_gpu(10), try_all_gpus()
    x = torch.tensor([1, 2, 3])
    print(x.device)
    X = torch.ones(2, 3, device=try_gpu())
    print(X)
    Y = torch.rand(2, 3, device=try_gpu(1))
    print(Y)
    Z = X.cuda(1)
    print(X)
    print(Z)
    print(Y + Z)
    print(Z.cuda(1) is Z)
    net = nn.Sequential(nn.Linear(3, 1))
    net = net.to(device=try_gpu())
    net(X)
    net[0].weight.data.device
