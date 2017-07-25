#!/usr/bin/env python
# -*- coding: utf8 -*-

import os.path
import sys
from matplotlib import pyplot

dir_name = os.path.abspath(os.path.dirname(__file__))
libs_path = os.path.join(dir_name, 'libs')
sys.path.insert(0, libs_path)

from meteorolData import MeteorolData
from batchGenerator import BatchGenerator

data = MeteorolData()
# Plot wind speed data
# speed = data.windSpeed.tolist()
# pyplot.plot(speed)
# pyplot.show()

data.toIOSet(iTimeStep=5)
print data.inputSet
print data.outputSet

batch = BatchGenerator(data.inputSet, data.outputSet, 3)
print batch.inputBatch
print batch.outputBatch
