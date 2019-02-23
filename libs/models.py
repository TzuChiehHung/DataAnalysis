from torch import nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, output_size=1, num_layers=1, batch_first=True):
        super(SimpleRNN, self).__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,     # rnn hidden unit
            num_layers=num_layers,       # number of rnn layer
            batch_first=batch_first,   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x, h_state):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        r_out, h_state = self.rnn(x, h_state)

        # outs = []    # save all predictions
        # for time_step in range(r_out.size(1)):    # calculate output for each time step
        #     outs.append(self.out(r_out[:, time_step, :]))
        # return torch.stack(outs, dim=1), h_state

        # instead, for simplicity, you can replace above codes by follows
        # r_out = r_out.view(-1, 32)
        # outs = self.out(r_out)
        # outs = outs.view(-1, TIME_STEP, 1)
        # return outs, h_state
        
        # or even simpler, since nn.Linear can accept inputs of any dimension 
        # and returns outputs with same dimension except for the last
        outs = self.out(r_out)
        return outs, h_state