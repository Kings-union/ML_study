import tensorflow as tf

# setup a linear classifier
import math
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from tensorflow.python.data import Dataset


tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

california_housing_dataframe = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")
california_housing_dataframe = california_housing_dataframe.reindex(np.random.permutation(california_housing_dataframe.index))

def preprocess_features(california_housing_dataframe):
    """
    Prepares input features from California housing data set.
    :param california_housing_dataframe:
            A Pandas DataFrame expected to contain data from
            the California housing data set.
    :return:
            A DataFrame that contains the features to be used for the model,
            including synthetic features.
    """
    selected_features = california_housing_dataframe[
        [
            "latitude",
            "longitude",
            "housing_media_age",
            "total_rooms",
            "total_bedrooms",
            "population",
            "households",
            "median_income"
        ]
    ]

