import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib import rnn
import numpy as np
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pandas as pd
import math

# os.chdir('/home/issac.wu')
# usdtwd= pd.read_csv('usdtwd.csv', usecols= [1,2])

def split_time_series(data, val_size=.2, test_size= .2):
    ntest = int(round(len(data)*(1 - test_size)))
    nval = int(round(len(data.iloc[:ntest])*(1 - test_size)))
    df_train, df_val, df_test = data.iloc[:nval], data.iloc[nval:ntest], data.iloc[ntest:]
    return df_train, df_val, df_test

def create_rnn_data(data, time_step, tscol= 1, labels=False):
    rnn_df = []
    for i in range(len(data)-time_step):
        if labels:
            try:
                rnn_df.append(data.iloc[i+time_step, tscol].as_matrix())
            except AttributeError:
                #print 'error'
                rnn_df.append([data.iloc[i+time_step, tscol]])
        else:
            data_ = data.iloc[ i:i+time_step, tscol].as_matrix()
            rnn_df.append(data_ if len(data_.shape) > 1 else [[j] for j in data_])
    return rnn_df

def prepare_rnn_data(data, time_step, val_size=0.2, test_size=0.2):
    df_train, df_val, df_test = split_time_series(data, val_size, test_size)
    train_x = create_rnn_data(df_train, time_step, labels=False)
    val_x = create_rnn_data(df_val, time_step, labels=False)
    test_x = create_rnn_data(df_test, time_step, labels=False)
    train_y = create_rnn_data(df_train, time_step, labels=True)
    val_y = create_rnn_data(df_val, time_step, labels=True)
    test_y = create_rnn_data(df_test, time_step, labels=True)
    return dict(train=train_x, val=val_x, test=test_x), dict(train=train_y, val=val_y, test=test_y)

def RNN(x, weights, biases, time_step, rnn_layer):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, time_step, 1)
    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(rnn_layer['steps'], forget_bias=.5, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    # Linear activation, using rnn inner loop last output
    return {'output':outputs, 'state':states, 'pred':tf.matmul(outputs[-1], weights['out']) + biases['out']}


def CalculatePred(pred, X, Y):
    pred_train = sess.run(pred['pred'], feed_dict={indVar: X['train'], depVar: Y['train']})
    pred_train = [val[0] for val in pred_train]
    pred_eval = sess.run(pred['pred'], feed_dict={indVar: X['val'], depVar: Y['val']})
    pred_eval = [val[0] for val in pred_eval]
    pred_test = sess.run(pred['pred'], feed_dict={indVar: X['test'], depVar: Y['test']})
    pred_test = [val[0] for val in pred_test]
    return {'train':pred_train, 'val':pred_eval, 'test':pred_test}

LOG_DIR = './ops_logs'
TIMESTEPS = 50
RNN_LAYERS = [{'steps': 10}, {'steps': TIMESTEPS, 'keep_prob': 0.5}]
DENSE_LAYERS = [2]
TRAINING_STEPS = 100
BATCH_SIZE = 30
PRINT_STEPS = TRAINING_STEPS / 10
LEARNING_RATE = 0.001 

# tf Graph input
indVar = tf.placeholder("float32", [None, TIMESTEPS, 1])
depVar = tf.placeholder("float32", [None, 1])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([RNN_LAYERS[0]['steps'], 1]))
}
biases = {
    'out': tf.Variable(tf.random_normal([1]))
}


pred = RNN(indVar, weights, biases, time_step= TIMESTEPS, rnn_layer= RNN_LAYERS[0])

# Define loss using mean squared error and optimizer
#loss = tf.losses.mean_squared_error(tf.log(pred['pred']), tf.log(depVar))
loss = tf.losses.mean_squared_error(pred['pred'], depVar)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

# Evaluate model
#accuracy = tf.reduce_mean(tf.square(tf.log(depVar)-tf.cast(tf.log(pred['pred']), tf.float32)))
accuracy = tf.reduce_mean(tf.square(depVar-tf.cast(pred['pred'], tf.float32)))

period = 90
start_date = datetime.datetime.strptime('2011-01-01', "%Y-%m-%d")
def real_mean(t, period):
    return math.sin(math.pi*t/period)

col1 = [ start_date + datetime.timedelta(t) for t in range(5000) ]
col2 = [ real_mean(t, period= period) for t in range(5000)] + np.random.normal(0, 0.5, [5000])
sim_sin =pd.DataFrame(data = {
        'record_date': col1, 
        'values': col2
        }) 
simX, simy = prepare_rnn_data(sim_sin.loc[:, ['record_date', 'values']], TIMESTEPS)
sim_train, sim_val, sim_test = split_time_series(sim_sin, val_size=.2, test_size= .2)
DT = {'train':sim_train, 'val':sim_val, 'test':sim_test}

# Initializing the variables
init = tf.global_variables_initializer() 

# Launch the graph
sess= tf.InteractiveSession()
sess.run(init)

step = 1
training_loss = []
# Keep training until reach max iterations
while (step+BATCH_SIZE) < len(simX['train']):
    train_x = simX['train'][(step+0):(step+BATCH_SIZE)]
    train_y = simy['train'][(step+0):(step+BATCH_SIZE)]
    # Run optimization op (backprop)
    tmp_loss, _ = sess.run([loss, optimizer], feed_dict={indVar: train_x, depVar: train_y})
    training_loss.append(tmp_loss)
    step += 1

print("Optimization Finished!")

acc2 = sess.run(accuracy, feed_dict={indVar: simX['val'], depVar: simy['val']})
print acc2
prediction_value2 = sess.run(pred['pred'], feed_dict={indVar: simX['val'], depVar: simy['val']})
temp = sess.run(pred, feed_dict={indVar: simX['val'][:1], depVar: simy['val'][:1]})
prediction_value2 = [val[0] for val in prediction_value2]


# Plot Training Data Performance
pred_value = CalculatePred(pred, simX, simy)

x = np.array([datetime.datetime.strptime(date_sring, "%Y-%m-%d") for date_sring in sim_sin['record_date']])
y = np.array(sim_sin['values'])
color_def = {'train':mcolors.CSS4_COLORS['red'], 'val':mcolors.CSS4_COLORS['olive'], 'test':mcolors.CSS4_COLORS['royalblue'] }
plt.style.use('fivethirtyeight')
fig = plt.figure(figsize=[16,9])
plt.plot_date(x, y, color='grey', linestyle='-', lw=.8, markersize=1, marker='.')
for dtset in pred_value:
    x = np.array([datetime.datetime.strptime(date_sring, "%Y-%m-%d") for date_sring in DT[dtset]['record_date'] ])
    plt.plot_date(x[TIMESTEPS:], np.log(pred_value[dtset]), fmt=color_def[dtset], lw=.8)

plt.title("RNN on USDTWD")
plt.ylabel("USDTWD")
plt.grid(True)
plt.show()
fig.savefig('RNN_USDTWD_train.png')   # save the figure to file
plt.close(fig)    # close the figure

fig = plt.figure(figsize=[16,9])
#plt.plot(x=range(len(training_loss)), y=training_loss, color=color_def['test'], linewidth=.8)
plt.xlim(-1, len(training_loss)+1)
plt.ylim(-1, max(training_loss))
plt.plot(range(len(training_loss)), training_loss, color='green', linestyle='-', linewidth=0.8, marker='.', markersize=1)
plt.title("RNN Training loss")
plt.ylabel("MSE")
plt.grid(True)
plt.show()
fig.savefig('RNN_USDTWD_training_loss.png')   # save the figure to file
plt.close(fig)    # close the figure


sess.close()



