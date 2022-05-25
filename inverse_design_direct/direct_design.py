import torch
from torch import nn
device = torch.device("cuda" if torch.cuda. is_available() else "cpu")

model_nn_1 = nn.Sequential(
    nn.Linear(6, 9),
    nn.Softmax(1),
    nn.Linear(9, 16),
    nn.Softmax(1),
    nn.Linear(16, 12),
    nn.Softmax(1)
).to(device).double()

model_nn_2 = nn.Sequential(
    nn.Linear(6,9),
    nn.Softmax(1),
    nn.Linear(9,12),
    nn.Softmax(1),
    nn.Linear(12,12),
    nn.Softmax(1)
).to(device).double()
criterion = nn.MSELoss()
loss_hist = []
R2_hist = []

def train_NN_1(x, y, max_iter, m=model_nn_1):
    optimizer = torch.optim.Adam(m.parameters())
    y_bar = torch.mean(y, 0)

    for t in range(1, max_iter + 1):
        y_pred = m(x)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if t % 100 == 0:
            loss_hist.append(loss.detach().cpu())
            R2 = 1 - torch.sum((y - y_pred) ** 2, 0) / torch.sum((y - y_bar) ** 2, 0)
            R2 = torch.mean(R2)
            R2_hist.append(R2.detach().cpu())
            print(f'epoch: {t}, loss: {float(loss.item()):.4f}, R^2: {float(R2):.4f}')
            if len(loss_hist) > 2 and torch.abs((loss_hist[-1] - loss_hist[-2]) / loss_hist[-1]) < 1e-4:
                break
    return m, loss_hist, R2_hist


def train_NN_2(x, y, max_iter, m=model_nn_2):
    optimizer = torch.optim.Adam(m.parameters())
    y_bar = torch.mean(y, 0)
    
    for t in range(1, max_iter + 1):
        y_pred = m(x)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if t % 100 == 0:
            loss_hist.append(loss.detach().cpu())
            R2 = 1 - torch.sum((y - y_pred) ** 2, 0) / torch.sum((y - y_bar) ** 2, 0)
            R2 = torch.mean(R2)
            R2_hist.append(R2.detach().cpu())
            print(f'epoch: {t}, loss: {float(loss.item()):.4f}, R^2: {float(R2):.4f}')
            if len(loss_hist) > 2 and torch.abs((loss_hist[-1] - loss_hist[-2]) / loss_hist[-1]) < 1e-4:
                break
    return m, loss_hist, R2_hist
