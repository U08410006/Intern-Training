import math
import torch
from torch import nn
from d2l import torch as d2l
from torch.nn import functional as F

BATCH_SIZE, NUM_STEPS = 32, 35
NUM_HIDDENS = 512


def get_params(vocab_size, num_hiddens, device):
    """
    initialize the model parameters for the RNN model
    """
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # Hidden layer parameters
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    # Output layer parameters
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # Attach gradients
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


def init_rnn_state(batch_size, num_hiddens, device):
    """
    initial the rnn model state
    """
    return (torch.zeros((batch_size, num_hiddens), device=device),)


def rnn(inputs, state, params):
    """
    define an RNN model
    """
    # Here `inputs` shape: (`num_steps`, `batch_size`, `vocab_size`)
    W_xh, W_hh, b_h, W_hq, b_q = params
    (H,) = state
    outputs = []
    # Shape of `X`: (`batch_size`, `vocab_size`)
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)


def predict_ch8(prefix, num_preds, net, vocab, device):
    """
    Generate new characters following the `prefix`.
    """
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]

    def get_input():
        return torch.tensor([outputs[-1]], device=device).reshape((1, 1))

    for y in prefix[1:]:  # Warm-up period
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # Predict `num_preds` steps
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return "".join([vocab.idx_to_token[i] for i in outputs])


def grad_clipping(net, theta):
    """
    Clip the gradient.
    """
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


def train_epoch_ch8(
    net, train_iter, loss_function, updater, device, use_random_iter
):
    """
    Train a net within one epoch (defined in Chapter 8).
    """
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # Sum of training loss, no. of tokens
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # Initialize `state` when either it is the first iteration or
            # using random sampling
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # `state` is a tensor for `nn.GRU`
                state.detach_()
            else:
                # `state` is a tuple of tensors for `nn.LSTM` and
                # for our custom scratch implementation
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        loss = loss_function(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            loss.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            loss.backward()
            grad_clipping(net, 1)
            # Since the `mean` function has been invoked
            updater(batch_size=1)
        metric.add(loss * y.numel(), y.numel())
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()


def train_ch8(
    net, train_iter, vocab, lr, num_epochs, device, use_random_iter=False
):
    """
    Train a model (defined in Chapter 8).
    """
    animator = d2l.Animator(
        xlabel="epoch",
        ylabel="perplexity",
        legend=["train"],
        xlim=[10, num_epochs],
    )

    def updater_initial(batch_size):
        return d2l.sgd(net.params, lr, batch_size)

    def predict(prefix):
        return predict_ch8(prefix, 50, net, vocab, device)

    # Initialize
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = updater_initial()

    # Train and predict
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net,
            train_iter,
            nn.CrossEntropyLoss,
            updater,
            device,
            use_random_iter,
        )
        if (epoch + 1) % 10 == 0:
            print(predict("time traveller"))
            animator.add(epoch + 1, [ppl])
    print(f"perplexity {ppl:.1f}, {speed:.1f} tokens/sec on {str(device)}")
    print(predict("time traveller"))
    print(predict("traveller"))


class RNNModelScratch:
    """
    A RNN Model implemented from scratch.
    """

    def __init__(
        self,
        vocab_size,
        num_hiddens,
        device,
        get_params,
        init_state,
        forward_fn,
    ):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)


if __name__ == "__main__":
    NUM_EPOCHS, LR = 500, 1
    train_iter, vocab = d2l.load_data_time_machine(BATCH_SIZE, NUM_STEPS)
    X = torch.arange(10).reshape((2, 5))
    print(F.one_hot(X.T, 28).shape)
    net = RNNModelScratch(
        len(vocab), NUM_HIDDENS, d2l.try_gpu(), get_params, init_rnn_state, rnn
    )
    state = net.begin_state(X.shape[0], d2l.try_gpu())
    Y, new_state = net(X.to(d2l.try_gpu()), state)
    Y.shape, len(new_state), new_state[0].shape
    predict_ch8("time traveller ", 10, net, vocab, d2l.try_gpu())
    train_ch8(net, train_iter, vocab, LR, NUM_EPOCHS, d2l.try_gpu())
    net = RNNModelScratch(
        len(vocab), NUM_HIDDENS, d2l.try_gpu(), get_params, init_rnn_state, rnn
    )
    train_ch8(
        net,
        train_iter,
        vocab,
        LR,
        NUM_EPOCHS,
        d2l.try_gpu(),
        use_random_iter=True,
    )
