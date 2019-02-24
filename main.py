import os.path
import sys
from argparse import ArgumentParser
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

# loss function
def loss_func(prediction, targets):
    criterion = torch.nn.MSELoss()
    return criterion(prediction, targets)

def train(model, train_dataloader, val_dataloader, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)   # optimize all rnn parameters
    # criterion = torch.nn.MSELoss()

    for epoch in range(args.epochs):
        tic = time()
        for i, (x, y) in enumerate(train_dataloader):
            x = x.type(torch.FloatTensor)
            y = y.type(torch.FloatTensor)
            h_state = None                          # initial hidden state
            y_hat, h_state = model(x, h_state)      # rnn model output
            # !! next step is important !!
            # h_state = h_state.data                # repack the hidden state, break the connection from last iteration

            train_loss = loss_func(y_hat, y)        # calculate loss
            optimizer.zero_grad()                   # clear gradients for this training step
            train_loss.backward()                   # backpropagation, compute gradients
            optimizer.step()                        # apply gradients
            # break

        val_loss = validation(model, val_dataloader)
        print('Epoch {:02d}: training_loss={:>8.5f}, val_loss={:>8.5f}, time={:>7.4f}s'
            .format(epoch, train_loss, val_loss, time() - tic))

def validation(model, val_dataloader):
    val_loss = 0
    for x, y in val_dataloader:
        with torch.no_grad():
            x = x.type(torch.FloatTensor)
            y = y.type(torch.FloatTensor)
            h_state = None

        y_hat, h_state = model(x, h_state)   # rnn model output
        val_loss += loss_func(y_hat, y)
    val_loss /= len(val_dataloader)
    return val_loss

def short_term_prediction(model, test_data, args):
    h_state = None
    prediction=[]
    for i in range(test_data.shape[0]-1):
        with torch.no_grad():
            x = torch.from_numpy(train_data[i][np.newaxis, :, np.newaxis]).type(torch.FloatTensor)
            y_hat, h_state = model(x, h_state)
            prediction.append(y_hat.item())

    targets = test_data[1:].flatten()
    prediction = np.array(prediction)
    error = targets - prediction

    print('\nShort-term prediction:')
    print(' MAE = {:.6f}'.format(np.abs(error).mean()))
    print('RMSE = {:.6f}'.format(np.sqrt(np.square(error).mean())))

    if args.visual:
        plot_prediction(prediction, targets, error, title='Short-term prediction')

def long_term_prediction(model, test_data, args):
    h_state = None
    prediction=[]
    for i in range(test_data.shape[0]-1):
        with torch.no_grad():
            if i < args.time_step:
                x = torch.from_numpy(train_data[i][np.newaxis, :, np.newaxis]).type(torch.FloatTensor)
            else:
                x = y_hat
            y_hat, h_state = model(x, h_state)
            prediction.append(y_hat.item())

    targets = test_data[1:].flatten()
    prediction = np.array(prediction)
    error = targets - prediction

    print('\nLong-term prediction:')
    print(' MAE = {:.6f}'.format(np.abs(error).mean()))
    print('RMSE = {:.6f}'.format(np.sqrt(np.square(error).mean())))

    if args.visual:
        plot_prediction(prediction, targets, error, title='Long-term prediction')

def plot_prediction(prediction, targets, error, title=''):
    _, ax = plt.subplots()
    ax.plot(prediction,'b-', label='predict value')
    ax.plot(targets, 'r-', label='historical data')
    ax.plot(np.abs(error), 'g-', label='error')
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax.legend()
    plt.title(title)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--visual', action='store_true', help='Show training data and/or testing result.')
    parser.add_argument('--train', action='store_true', help='Train the model on training dataset.')
    parser.add_argument('--test', action='store_true', help='Test the trained model on testing dataset.')

    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--time_step', default=90, type=int)
    parser.add_argument('--hidden_size', default=32, type=int)
    parser.add_argument('--lr', default=0.02, type=float)
    args = parser.parse_args()

    # data pre-processing
    met = MetData()

    # load dataset
    met.split()
    train_data = met.get_training_data(['speedAvg'])
    val_data = met.get_val_data()

    train_dataset = SeqDataset(train_data, time_step=args.time_step)
    val_dataset = SeqDataset(val_data, time_step=args.time_step)

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=1)

    if args.visual:
        date = met.train.index.tolist()
        speed = met.train.speedAvg.tolist()
        _, ax = plt.subplots()
        ax.plot_date(date, speed, '-')
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
        plt.title('Training Data')
        plt.draw()

    # load model and weights
    if args.train or args.test:
        model = SimpleRNN(input_size=train_data.shape[1], hidden_size=args.hidden_size)
        model.eval()
        print('\n' + str(model) + '\n')

    if args.train:
        train(model, train_dataloader, val_dataloader, args)

    if args.test:
        short_term_prediction(model, val_data, args)
        long_term_prediction(model, val_data, args)

    if args.visual:
        plt.show()
