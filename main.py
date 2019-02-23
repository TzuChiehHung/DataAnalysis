#%%
import os.path
import sys
from matplotlib import dates as mdates
from matplotlib import pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from libs.meteorolData import MeteorolData
from libs.batchGenerator import BatchGenerator

#%% data processing
meteoro = MeteorolData()

#%% split dataset for training and testing
meteoro.split()
train = meteoro.get_training_data()
val = meteoro.get_val_data()

# Plot
date = meteoro.val.index.tolist()
speed = meteoro.val.speedAvg.tolist()
plt.plot_date(date, speed, '-')
# set ticks every week
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gca().xaxis.set_minor_locator(mdates.MonthLocator())
plt.show()

# data.toIOSet(iTimeStep=5)
# print(data.inputSet)
# print(data.outputSet)

# batch = BatchGenerator(data.inputSet, data.outputSet, 3)
# print(batch.inputBatch)
# print(batch.outputBatch)

