import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


class RNNModel(nn.Module):
    """The RNN model."""

    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        # If the RNN is bidirectional (to be introduced later),
        # `num_directions` should be 2, else it should be 1.
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        # The fully connected layer will first change the shape of `Y` to
        # (`num_steps` * `batch_size`, `num_hiddens`). Its output shape is
        # (`num_steps` * `batch_size`, `vocab_size`).
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # `nn.GRU` takes a tensor as hidden state
            return torch.zeros(
                (
                    self.num_directions * self.rnn.num_layers,
                    batch_size,
                    self.num_hiddens,
                ),
                device=device,
            )
        else:
            # `nn.LSTM` takes a tuple of hidden states
            return (
                torch.zeros(
                    (
                        self.num_directions * self.rnn.num_layers,
                        batch_size,
                        self.num_hiddens,
                    ),
                    device=device,
                ),
                torch.zeros(
                    (
                        self.num_directions * self.rnn.num_layers,
                        batch_size,
                        self.num_hiddens,
                    ),
                    device=device,
                ),
            )


if __name__ == "__main__":
    BARCH_SIZE, NUM_STEPS = 32, 35
    NUM_HIDDENS = 256
    NUM_EPOCHS, LR = 500, 1
    train_iter, vocab = d2l.load_data_time_machine(BARCH_SIZE, NUM_STEPS)
    rnn_layer = nn.RNN(len(vocab), NUM_HIDDENS)
    state = torch.zeros((1, BARCH_SIZE, NUM_HIDDENS))
    print(state.shape)
    X = torch.rand(size=(NUM_STEPS, BARCH_SIZE, len(vocab)))
    Y, state_new = rnn_layer(X, state)
    print(Y.shape, state_new.shape)
    device = d2l.try_gpu()
    net = RNNModel(rnn_layer, vocab_size=len(vocab))
    net = net.to(device)
    d2l.predict_ch8("time traveller", 10, net, vocab, device)
    d2l.train_ch8(net, train_iter, vocab, LR, NUM_EPOCHS, device)
