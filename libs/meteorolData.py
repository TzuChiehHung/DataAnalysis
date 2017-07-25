#!/usr/bin/env python
# -*- coding: utf8 -*-

import csv
import pandas as pd


class MeteorolData():

    def __init__(self, filename = 'data/TainanMeteorolData.csv'):
        self.filename = filename

        # load rawData
        self.rawData = pd.read_csv(self.filename, sep='\t', skiprows=0)
        self.rawData.columns = ['station', 'date', 'tempAvg', 'speedAvg', 'directAvg',
                              'sunRate', 'solarIrr']
        self.windSpeed()

    def windSpeed(self):
        rawData = self.rawData
        # interpolation missed rawData
        index = rawData[rawData['speedAvg']<0].index
        for i in index:
            rawData['speedAvg'][i] = (rawData['speedAvg'][i-1] + rawData['speedAvg'][i+1])/2

        # group 2 station rawData by date
        self.windSpeed = rawData.groupby(['date'])['speedAvg'].mean()

    def toIOSet(self, iTimeStep, oTimeStep=1):
        timeSeries = self.windSpeed.tolist()
        timeSeries = range(1, 20) # testing data
        self.inputSet = []
        self.outputSet = []
        for i in range(len(timeSeries)-iTimeStep-oTimeStep+1):
            self.inputSet.append(timeSeries[i:i+iTimeStep])
            self.outputSet.append(timeSeries[i+iTimeStep:i+iTimeStep+oTimeStep])
