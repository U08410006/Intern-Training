import torch
from torch import nn  # `nn` is an abbreviation for neural networks
from torch.utils import data
from d2l import torch as d2l


def load_array(data_arrays, batch_size, is_train=True):
    """
    Construct a PyTorch data iterator.
    """
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


if __name__ == "__main__":
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = d2l.synthetic_data(true_w, true_b, 1000)

    BATCH_SIZE = 10
    data_iter = load_array((features, labels), BATCH_SIZE)

    print(next(iter(data_iter)))

    net = nn.Sequential(nn.Linear(2, 1))

    net[0].weight.data.normal_(0, 0.01)
    print(net[0].bias.data.fill_(0))

    calculate_loss = nn.MSELoss()

    trainer = torch.optim.SGD(net.parameters(), lr=0.03)

    num_epochs = 3
    for epoch in range(num_epochs):
        for X, y in data_iter:
            loss = calculate_loss(net(X), y)
            trainer.zero_grad()
            loss.backward()
            trainer.step()
        loss = calculate_loss(net(features), labels)
        print(f"epoch {epoch + 1}, loss {loss:f}")

    w = net[0].weight.data
    print("error in estimating w:", true_w - w.reshape(true_w.shape))
    b = net[0].bias.data
    print("error in estimating b:", true_b - b)
