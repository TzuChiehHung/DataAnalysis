#!/usr/bin/env python
# -*- coding: utf8 -*-

class BatchGenerator():

    def __init__(self, inputSet, outputSet, batchSize):
        self.input = inputSet
        self.output = outputSet
        self.batchSize = batchSize
        self.batches = len(inputSet) // batchSize

        self.inputBatch = []
        self.outputBatch = []

        for i in range(self.batches):
            self.inputBatch.append(inputSet[i:i+self.batchSize])
            self.outputBatch.append(outputSet[i:i+self.batchSize])
