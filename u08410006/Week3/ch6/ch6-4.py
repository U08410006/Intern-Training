import torch
from d2l import torch as d2l


def corr2d_multi_in(input_tensor, kernel_tensor):
    """
    First, iterate through the 0th dimension (channel dimension) of `X` and
    `K`. Then, add them together
    """
    return sum(d2l.corr2d(x, k) for x, k in zip(input_tensor, kernel_tensor))


def corr2d_multi_in_out(input_tensor, kernel_tensor):
    """
    Iterate through the 0th dimension of `K`, and each time, perform
    cross-correlation operations with input `X`. All of the results are
    stacked together
    """
    return torch.stack(
        [corr2d_multi_in(input_tensor, k) for k in kernel_tensor], 0
    )


def corr2d_multi_in_out_1x1(input_tensor, kernel_tensor):
    """
    implement a 1×1 convolution using a fully-connected layer.
    """
    channel_input_nums, height, width = input_tensor.shape
    channel_output_nums = kernel_tensor.shape[0]
    input_new = input_tensor.reshape((channel_input_nums, height * width))
    Kernel_new = kernel_tensor.reshape(
        (channel_output_nums, channel_input_nums)
    )
    # Matrix multiplication in the fully-connected layer
    output_tensor = torch.matmul(Kernel_new, input_new)
    return output_tensor.reshape((channel_output_nums, height, width))


if __name__ == "__main__":
    X = torch.tensor(
        [
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        ]
    )
    K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])

    print(corr2d_multi_in(X, K))
    X = torch.tensor(
        [
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        ]
    )
    K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])

    print(corr2d_multi_in(X, K))
    K = torch.stack((K, K + 1, K + 2), 0)
    print(K.shape)
    print(corr2d_multi_in_out(X, K))
    X = torch.normal(0, 1, (3, 3, 3))
    K = torch.normal(0, 1, (2, 3, 1, 1))

    Y1 = corr2d_multi_in_out_1x1(X, K)
    Y2 = corr2d_multi_in_out(X, K)
    assert float(torch.abs(Y1 - Y2).sum()) < 1e-6
