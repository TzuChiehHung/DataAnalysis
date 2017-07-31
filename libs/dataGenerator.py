#!/usr/bin/env python
# -*- coding: utf8 -*-

# Generate data with size (nData, nTimeStep)

import math
import numpy as np

class DataGenerator():

    def __init__(self, nTimeStep, nData = 1, basePeriod = 50):
        self.nTimeStep = nTimeStep
        self.nData = nData
        self.time = range(self.nTimeStep)

        self.mean = []
        for scale in range(1, self.nData+1):
            xi = [math.sin(2*math.pi*t/(scale*basePeriod)) for t in self.time]
            self.mean.append(xi)
        self.mean = np.asarray(self.mean)

        self.data = self.mean + np.random.normal(
            0, 1, [self.nData, self.nTimeStep])

    # Generate input to output data set
    # Generate a list of ndarray with shape (nData, iTimeStep)
    def toIOSet(self, iTimeStep, oTimeStep=1):

        self.inputSet = []
        self.outputSet = []

        for i in range(self.nTimeStep-iTimeStep-oTimeStep+1):
            self.inputSet.append(self.data[:,i:i+iTimeStep])
            self.outputSet.append(self.data[:,i+iTimeStep:i+iTimeStep+oTimeStep])
