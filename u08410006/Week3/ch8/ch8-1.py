import torch
from torch import nn
from d2l import torch as d2l


def init_weights(m):
    """
    Function for initializing the weights of the network
    """
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


def get_net():
    """
    A simple MLP
    """
    net = nn.Sequential(nn.Linear(4, 10), nn.ReLU(), nn.Linear(10, 1))
    net.apply(init_weights)
    return net


def train(net, train_iter, loss_function, epochs, lr):
    """
    define how to train
    """
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            loss = loss_function(net(X), y)
            loss.backward()
            trainer.step()
        print(
            f"epoch {epoch + 1}, "
            f"loss: {d2l.evaluate_loss(net, train_iter, loss_function):f}"
        )


if __name__ == "__main__":
    TOTAL = 1000  # Generate a total of 1000 points
    TAU = 4
    BATCH_SIZE, N_TRAIN = 16, 600
    MAX_STEPS = 64
    STEPS = (1, 4, 16, 64)
    time = torch.arange(1, TOTAL + 1, dtype=torch.float32)
    x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (TOTAL,))
    d2l.plot(time, [x], "time", "x", xlim=[1, 1000], figsize=(6, 3))

    features = torch.zeros((TOTAL - TAU, TAU))
    for i in range(TAU):
        features[:, i] = x[i : TOTAL - TAU + i]
    labels = x[TAU:].reshape((-1, 1))

    # Only the first `n_train` examples are used for training
    train_iter = d2l.load_array(
        (features[:N_TRAIN], labels[:N_TRAIN]), BATCH_SIZE, is_train=True
    )

    train(get_net, train_iter, nn.MSELoss, 5, 0.01)
    onestep_preds = get_net(features)
    d2l.plot(
        [time, time[TAU:]],
        [x.detach().numpy(), onestep_preds.detach().numpy()],
        "time",
        "x",
        legend=["data", "1-step preds"],
        xlim=[1, 1000],
        figsize=(6, 3),
    )
    multistep_preds = torch.zeros(TOTAL)
    multistep_preds[: N_TRAIN + TAU] = x[: N_TRAIN + TAU]
    for i in range(N_TRAIN + TAU, TOTAL):
        multistep_preds[i] = get_net(
            multistep_preds[i - TAU : i].reshape((1, -1))
        )

    d2l.plot(
        [time, time[TAU:], time[N_TRAIN + TAU :]],
        [
            x.detach().numpy(),
            onestep_preds.detach().numpy(),
            multistep_preds[N_TRAIN + TAU :].detach().numpy(),
        ],
        "time",
        "x",
        legend=["data", "1-step preds", "multistep preds"],
        xlim=[1, 1000],
        figsize=(6, 3),
    )

    multistep_preds = torch.zeros(TOTAL)
    multistep_preds[: N_TRAIN + TAU] = x[: N_TRAIN + TAU]
    for i in range(N_TRAIN + TAU, TOTAL):
        multistep_preds[i] = get_net(
            multistep_preds[i - TAU : i].reshape((1, -1))
        )

    d2l.plot(
        [time, time[TAU:], time[N_TRAIN + TAU :]],
        [
            x.detach().numpy(),
            onestep_preds.detach().numpy(),
            multistep_preds[N_TRAIN + TAU :].detach().numpy(),
        ],
        "time",
        "x",
        legend=["data", "1-step preds", "multistep preds"],
        xlim=[1, 1000],
        figsize=(6, 3),
    )

    features = torch.zeros((TOTAL - TAU - MAX_STEPS + 1, TAU + MAX_STEPS))
    # Column `i` (`i` < `tau`) are observations from `x` for time steps from
    # `i + 1` to `i + T - tau - max_steps + 1`
    for i in range(TAU):
        features[:, i] = x[i : i + TOTAL - TAU - MAX_STEPS + 1]

    # Column `i` (`i` >= `tau`) are the (`i - tau + 1`)-step-ahead predictions for
    # time steps from `i + 1` to `i + T - tau - max_steps + 1`
    for i in range(TAU, TAU + MAX_STEPS):
        features[:, i] = TAU(features[:, i - TAU : i]).reshape(-1)

    d2l.plot(
        [time[TAU + i - 1 : TOTAL - MAX_STEPS + i] for i in STEPS],
        [features[:, (TAU + i - 1)].detach().numpy() for i in STEPS],
        "time",
        "x",
        legend=[f"{i}-step preds" for i in STEPS],
        xlim=[5, 1000],
        figsize=(6, 3),
    )
