import torch
from torch import nn


def comp_conv2d(conv2d, input_tensor):
    """
    We define a convenience function to calculate the convolutional layer. This
    function initializes the convolutional layer weights and performs
    corresponding dimensionality elevations and reductions on the input and
    output
    """
    # Here (1, 1) indicates that the batch size and the number of channels
    # are both 1
    channel_one = input_tensor.reshape((1, 1) + input_tensor.shape)
    output_tensor = conv2d(channel_one)
    # Exclude the first two dimensions that do not interest us: examples and
    # channels
    return output_tensor.reshape(output_tensor.shape[2:])


if __name__ == "__main__":
    # Note that here 1 row or column is padded on either side, so a total of 2
    # rows or columns are added
    conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
    X = torch.rand(size=(8, 8))
    print(comp_conv2d(conv2d, X).shape)
    # Here, we use a convolution kernel with a height of 5 and a width of 3. The
    # padding numbers on either side of the height and width are 2 and 1,
    # respectively
    conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))
    print(comp_conv2d(conv2d, X).shape)
    conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
    print(comp_conv2d(conv2d, X).shape)
    conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
    print(comp_conv2d(conv2d, X).shape)
