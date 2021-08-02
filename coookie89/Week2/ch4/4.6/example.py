# 4.6. Dropout

import torch
from torch import nn
from d2l import torch as d2l


def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1

    # In this case, all elements are dropped out
    if dropout == 1:
        return torch.zeros_like(X)

    # In this case, all elements are kept
    if dropout == 0:
        return X

    mask = (torch.rand(X.shape) > dropout).float()
    return mask * X / (1.0 - dropout)


# Defining the Model
class Net(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_outputs,
        num_hiddens1,
        num_hiddens2,
        dropout1,
        dropout2,
        is_training=True,
    ):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()
        self.dropout1 = dropout1
        self.dropout2 = dropout2

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))

        # Use dropout only when training the model
        if self.training is True:
            # Add a dropout layer after the first fully connected layer
            H1 = dropout_layer(H1, self.dropout1)
        H2 = self.relu(self.lin2(H1))

        if self.training is True:
            # Add a dropout layer after the second fully connected layer
            H2 = dropout_layer(H2, self.dropout2)
        out = self.lin3(H2)

        return out


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


# Training and Testing
if __name__ == "__main__":

    # Defining Model Parameters
    num_inputs = 784
    num_outputs = 10
    num_hiddens1 = 256
    num_hiddens2 = 256

    dropout1 = 0.2
    dropout2 = 0.2

    print("Training with complecated step-----------------------\n")
    net = Net(
        num_inputs, num_outputs, num_hiddens1, num_hiddens2, dropout1, dropout2
    )

    num_epochs = 10
    lr = 0.5
    batch_size = 256

    loss = nn.CrossEntropyLoss()
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

    print("Training with concise implementation-----------------\n")
    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(num_inputs, num_hiddens1),
        nn.ReLU(),
        # Add a dropout layer after the first fully connected layer
        nn.Dropout(dropout1),
        nn.Linear(num_hiddens1, num_hiddens2),
        nn.ReLU(),
        # Add a dropout layer after the second fully connected layer
        nn.Dropout(dropout2),
        nn.Linear(num_hiddens2, num_outputs),
    )

    net.apply(init_weights)

    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
