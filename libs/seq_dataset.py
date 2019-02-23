from torch.utils.data import Dataset

class SeqDataset(Dataset):


    def __init__(self, data, time_step=1, time_gap=1):
        self.data = data
        self._time_step = time_step
        self._time_gap = time_gap
        self._len = self.data.shape[0] - self._time_step - self._time_gap + 1

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        x = self.data[idx:idx+self._time_step]
        y = self.data[idx+self._time_gap:idx+self._time_step+self._time_gap]
        return x, y 
