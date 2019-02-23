import csv
import pandas as pd

class MeteorolData():

    def __init__(self, filename = 'data/TainanMeteorolData.csv'):
        self.filename = filename

        # load rawData
        self.raw = pd.read_csv(self.filename, sep='\t', skiprows=0)
        self.raw.columns = ['station', 'date', 'tempAvg', 'speedAvg', 'directAvg',
                              'sunRate', 'solarIrr']

        self.data = self.data_preproc()
        print(self.data.columns.tolist())

        self.split_date = '2007-01-01'
        self.train = None
        self.val = None

    def get_training_data(self, col_names='speedAvg'):
        if self.train is None:
            self.split()
        return self.train[col_names].values

    def get_val_data(self, col_names='speedAvg'):
        if self.val is None:
            self.split()
        return self.val[col_names].values

    def split(self, split_date=None):
        if split_date:
            self.split_date = split_date

        self.train = self.data[self.data.index < self.split_date]
        self.val = self.data[self.data.index >= self.split_date]

    def data_preproc(self):
        raw = self.raw

        # calculate mean value
        for col_name in raw.columns[2:]:
            avg = raw[raw[col_name]>0][col_name].mean()

            # replace missed data with yearly mean value
            idx = raw[raw[col_name]<0].index
            raw.loc[idx, col_name] = avg

        # group 2 station rawData by date
        data = raw.groupby(['date'])[raw.columns[2:]].mean()

        # convert date to timestamp
        data.index = pd.to_datetime(data.index.astype(int).map(str))

        return data

    # def toIOSet(self, iTimeStep, oTimeStep=1):
    #     timeSeries = self.windSpeed.tolist()
    #     timeSeries = range(1, 20) # testing data
    #     self.inputSet = []
    #     self.outputSet = []
    #     for i in range(len(timeSeries)-iTimeStep-oTimeStep+1):
    #         self.inputSet.append(timeSeries[i:i+iTimeStep])
    #         self.outputSet.append(timeSeries[i+iTimeStep:i+iTimeStep+oTimeStep])
