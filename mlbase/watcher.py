import h5py

class Watcher():
    def __init__(self):
        pass

    def toDataFrame(self):
        """
        Convert the collected data into pandas DataFrame
        """
        pass

    def replace(self, key, data):
        """
        Replace data with key squence with new data.
        """
        pass

    def append(self, key, data, jumpIndex=None):
        """
        Append new data to sequence of data under key
        """
        pass
