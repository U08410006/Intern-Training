import torch
from torch import nn


def pool2d(input_tensor, pool_size, mode="max"):
    """
    do Maximum Pooling and Average Pooling operation in 2d
    """
    pooling_height, pooling_weight = pool_size
    output = torch.zeros(
        (
            input_tensor.shape[0] - pooling_height + 1,
            input_tensor.shape[1] - pooling_weight + 1,
        )
    )
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            if mode == "max":
                output[i, j] = input_tensor[
                    i : i + pooling_height, j : j + pooling_weight
                ].max()
            elif mode == "avg":
                output[i, j] = input_tensor[
                    i : i + pooling_height, j : j + pooling_weight
                ].mean()
    return output


if __name__ == "__main__":
    X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    print(pool2d(X, (2, 2)))
    print(pool2d(X, (2, 2), "avg"))
    X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
    print(X)
    pool2d = nn.MaxPool2d(3)
    print(pool2d(X))
    pool2d = nn.MaxPool2d(3, padding=1, stride=2)
    print(pool2d(X))
    pool2d = nn.MaxPool2d((2, 3), stride=(2, 3), padding=(0, 1))
    print(pool2d(X))
    X = torch.cat((X, X + 1), 1)
    print(X)
    pool2d = nn.MaxPool2d(3, padding=1, stride=2)
    print(pool2d(X))
