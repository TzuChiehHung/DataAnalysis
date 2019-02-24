import os.path
import sys
from matplotlib import dates as mdates
from matplotlib import pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import torch
from torch.utils.data import DataLoader
import numpy as np
from time import time

from libs.met_data import MetData
from libs.seq_dataset import SeqDataset
from libs.models import SimpleRNN

# hyper-parameters
EPOCHS = 1
TIME_STEP = 90      # rnn time step
INPUT_SIZE = 1      # rnn input size
HIDDEN_SIZE = 32    # rnn hidden size
LR = 0.02           # learning rate

# data pre-processing
met = MetData()

# load dataset
met.split()
train_data = met.get_training_data(['speedAvg'])
val_data = met.get_val_data()

train_dataset = SeqDataset(train_data, time_step=TIME_STEP)
val_dataset = SeqDataset(val_data, time_step=TIME_STEP)

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=1)

# plotting
# plt.figure()
# date = met.val.index.tolist()
# speed = met.val.speedAvg.tolist()
# plt.plot_date(date, speed, '-')
# # set ticks every month
# plt.gca().xaxis.set_major_locator(mdates.YearLocator())
# plt.gca().xaxis.set_minor_locator(mdates.MonthLocator())

# model
model = SimpleRNN(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE)
print(model)

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LR)   # optimize all rnn parameters
criterion = torch.nn.MSELoss()

# validation
def validation(model, val_dataloader):
    val_loss = 0
    for x, y in val_dataloader:
        with torch.no_grad():
            x = x.type(torch.FloatTensor)
            y = y.type(torch.FloatTensor)
            h_state = None

        y_hat, h_state = model(x, h_state)   # rnn model output
        val_loss += criterion(y_hat, y)
    val_loss /= len(val_dataloader)
    return val_loss

# training
for epoch in range(EPOCHS):
    tic = time()
    for i, (x, y) in enumerate(train_dataloader):
        x = x.type(torch.FloatTensor)
        y = y.type(torch.FloatTensor)
        h_state = None                          # initial hidden state
        y_hat, h_state = model(x, h_state)      # rnn model output
        # !! next step is important !!
        # h_state = h_state.data                # repack the hidden state, break the connection from last iteration

        train_loss = criterion(y_hat, y)        # calculate loss
        optimizer.zero_grad()                   # clear gradients for this training step
        train_loss.backward()                   # backpropagation, compute gradients
        optimizer.step()                        # apply gradients

    val_loss = validation(model, val_dataloader)
    print('Epoch {:02d}: training_loss={:>8.5f}, val_loss={:>8.5f}, time={:>7.4f}s'
        .format(epoch, train_loss, val_loss, time() - tic))

# testing
h_state = None
# short-term preidction
Y=[]
for i in range(val_data.shape[0]-1):
    with torch.no_grad():
        x = torch.from_numpy(train_data[i][np.newaxis, :, np.newaxis]).type(torch.FloatTensor)
        y_hat, h_state = model(x, h_state)
        Y.append(y_hat.item())

y = val_data[1:].flatten()
y_hat = np.array(Y)
error = y-y_hat

print('\nShort-term prediction:')
print(' MAE = {:.6f}'.format(np.abs(error).mean()))
print('RMSE = {:.6f}'.format(np.sqrt(np.square(error).mean())))

# plt.figure()
_, ax = plt.subplots()
ax.plot(y, 'r-', label='historical data')
ax.plot(y_hat,'b-', label='predict value')
ax.plot(np.abs(error), 'g-', label='error')
ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax.legend()
plt.show()

# if __name__ == '__main__':
#     parser = ArgumentParser()
#     parser.add_argument('-v', '--visual', action='store_true', help='show image frame')

#     args = parser.parse_args()

#     main(args)