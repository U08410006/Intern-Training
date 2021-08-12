import torch
from torch import nn


def corr2d(input_tensor, kernel_tensor):
    """Compute 2D cross-correlation."""
    h, w = kernel_tensor.shape
    output_tensor = torch.zeros(
        (input_tensor.shape[0] - h + 1, input_tensor.shape[1] - w + 1)
    )
    for i in range(output_tensor.shape[0]):
        for j in range(output_tensor.shape[1]):
            output_tensor[i, j] = (
                input_tensor[i : i + h, j : j + w] * kernel_tensor
            ).sum()
    return output_tensor


class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias


if __name__ == "__main__":
    X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    print(corr2d(X, K))
    X = torch.ones((6, 8))
    X[:, 2:6] = 0
    print(X)
    K = torch.tensor([[1.0, -1.0]])
    Y = corr2d(X, K)
    print(Y)
    print(corr2d(X.t(), K))
    # Construct a two-dimensional convolutional layer with 1 output channel and a
    # kernel of shape (1, 2). For the sake of simplicity, we ignore the bias here
    conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)

    # The two-dimensional convolutional layer uses four-dimensional input and
    # output in the format of (example, channel, height, width), where the batch
    # size (number of examples in the batch) and the number of channels are both 1
    X = X.reshape((1, 1, 6, 8))
    Y = Y.reshape((1, 1, 6, 7))
    lr = 3e-2  # Learning rate

    for i in range(10):
        Y_hat = conv2d(X)
        loss = (Y_hat - Y) ** 2
        conv2d.zero_grad()
        loss.sum().backward()
        # Update the kernel
        conv2d.weight.data[:] -= lr * conv2d.weight.grad
        if (i + 1) % 2 == 0:
            print(f"batch {i + 1}, loss {loss.sum():.3f}")

    print(conv2d.weight.data.reshape((1, 2)))
