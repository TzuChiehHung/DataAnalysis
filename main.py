# %%
import os.path
import sys
from matplotlib import dates as mdates
from matplotlib import pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import torch
from torch.utils.data import DataLoader
import numpy as np

from libs.met_data import MetData
from libs.seq_dataset import SeqDataset
from libs.models import SimpleRNN

# %% hyper-parameters
TIME_STEP = 90      # rnn time step
INPUT_SIZE = 1      # rnn input size
HIDDEN_SIZE = 32    # rnn hidden size
LR = 0.02           # learning rate

# %% data pre-processing
met = MetData()

# %% prepare dataset
met.split()
train = met.get_training_data()
val = met.get_val_data()

train_dataset = SeqDataset(train, time_step=TIME_STEP)
val_dataset = SeqDataset(val, time_step=TIME_STEP)

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=1)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

# plotting
plt.figure(1)
date = met.val.index.tolist()
speed = met.val.speedAvg.tolist()
plt.plot_date(date, speed, '-')
# set ticks every week
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gca().xaxis.set_minor_locator(mdates.MonthLocator())

# %% define model
model = SimpleRNN(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE)
print(model)

# %% train
optimizer = torch.optim.Adam(model.parameters(), lr=LR)   # optimize all rnn parameters
loss_func = torch.nn.MSELoss()

plt.figure(2, figsize=(12, 5))
plt.ion()           # continuously plot

for x, y in train_dataloader:
    x = x.type(torch.FloatTensor)
    y = y.type(torch.FloatTensor)
    h_state = None                          # initial hidden state
    y_hat, h_state = model(x, h_state)      # rnn model output
    # !! next step is important !!
    # h_state = h_state.data                # repack the hidden state, break the connection from last iteration

    loss = loss_func(y_hat, y)              # calculate loss
    optimizer.zero_grad()                   # clear gradients for this training step
    loss.backward()                         # backpropagation, compute gradients
    optimizer.step()                        # apply gradients

    # plotting
    plt.clf()
    plt.plot(y.data.numpy().flatten(), 'r-')
    plt.plot(y_hat.data.numpy().flatten(), 'b-')
    plt.draw()
    plt.pause(0.01)

# %% validation TODO
plt.figure(3)
for x, y in val_dataloader:
    x = x.type(torch.FloatTensor)
    y = y.type(torch.FloatTensor)
    h_state = None
    y_hat, h_state = model(x, h_state)   # rnn model output

    # plotting
    plt.clf()
    plt.plot(y.data.numpy().flatten(), 'r-')
    plt.plot(y_hat.data.numpy().flatten(), 'b-')
    plt.plot(np.abs(y.data.numpy().flatten()-y_hat.data.numpy().flatten()), 'g-')
    plt.draw()
    plt.pause(0.01)

plt.ioff()
plt.show()

# if __name__ == '__main__':
#     parser = ArgumentParser()
#     parser.add_argument('-v', '--visual', action='store_true', help='show image frame')

#     args = parser.parse_args()

#     main(args)