#!/usr/bin/env python
# -*- coding: utf8 -*-

#%%
import os.path
import sys
from matplotlib import pyplot

from libs.meteorolData import MeteorolData
from libs.batchGenerator import BatchGenerator

#%% data processing
data = MeteorolData()

# Plot wind speed data
speed = data.windSpeed.tolist()
pyplot.plot(speed)
pyplot.show()

data.toIOSet(iTimeStep=5)
print(data.inputSet)
print(data.outputSet)

batch = BatchGenerator(data.inputSet, data.outputSet, 3)
print(batch.inputBatch)
print(batch.outputBatch)
