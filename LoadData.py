import pandas as pd
import numpy as np


def load_data():
    """ Loads data from txt file into pandas frame and performs pre-processing.

    :return: data: Preprocessed pandas data frame
    """

    # Load data
    data = pd.read_csv('price.data', sep=",", header=None)
    data.columns = ['symbol', 'normalized loss', 'wheel base', 'length', 'width', 'height', 'curb weight',
                    'engine size', 'bore', 'stroke', 'compression ratio', 'horespower', 'peak rpm',
                    'city mpg', 'highway mpg', 'price']

    # Normalize data
    data = data.to_numpy()
    data[:, 1:] = (data[:, 1:] - data[:, :1].mean())/data[:, :1].std()
    data = pd.DataFrame(data)
    return data


load_data()

