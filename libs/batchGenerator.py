#!/usr/bin/env python
# -*- coding: utf8 -*-

import numpy as np

class BatchGenerator():

    def __init__(self, inputSet, outputSet, batchSize):
        self.input = inputSet
        self.output = outputSet
        self.batchSize = batchSize
        self.batches = len(inputSet) // batchSize

        self.inputBatch = []
        self.outputBatch = []
        self.currentBatch = 0

        for i in range(self.batches):
            self.inputBatch.append(inputSet[i*self.batchSize:(i+1)*self.batchSize])
            self.outputBatch.append(outputSet[i*self.batchSize:(i+1)*self.batchSize])

    def nextBatch(self):
        index = self.currentBatch % self.batches
        self.currentBatch += 1
        # return ndarray with shape (batchSize, nData, iTimeStep)
        return np.asarray(self.inputBatch[index]), np.asarray(self.outputBatch[index])
