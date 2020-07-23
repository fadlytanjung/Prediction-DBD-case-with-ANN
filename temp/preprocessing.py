import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import Imputer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import SGD, Adagrad, RMSprop, Adadelta, Adamax, Adam
from tensorflow.keras.models import model_from_json
import math

Imputer = SimpleImputer

class Preprocessing:
    def __init__(self):
        self.trainScore = 0
        self.testScore = 0
        self.rmseTrain = 0
        self.rmseTest = 0
        self.X = None
        self.y = None

    #load data input
    def load_data(self, file):
        data = pd.read_excel(file)
        return data.values
    
    def sum_data(self, data):
        df = pd.DataFrame(data)
        new_data = np.zeros(4, )
        new_data = data[:,[1,2]]
        return new_data
    
    #hapus kolom yang tidak digunakan
    def hapus_kolom(self, data, kolom):
        return np.delete(data, kolom, axis=1)
    
    def one_hot(self, data):
        values = data[:, 1]
        # define example
        #encode kecamatan ke integer
        # integer encode
        self.label_encoder = LabelEncoder()
        integer_encoded = self.label_encoder.fit_transform(values)
        data[:, 1] = integer_encoded
        return data
    
    def isi_kolom_kosong(self,data):
        values = data[:]
        self.imp = Imputer(missing_values=np.NAN, strategy='mean', fill_value=None, verbose=0, copy=True)
        fill_column = self.imp.fit_transform(values)
        data[:] = fill_column
        return data
    
    def split_data(self, data):
        #split data into X and Y variables
        X = data[:,[1,2,3,4,5,6,7,8]]
        y = data[:,0]
        y = y.reshape(-1,1)
        self.X = X
        self.y = y
        return X, y
    
    def normalisasi(self, data):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        X_scaled = self.scaler.fit_transform(self.X)
        y_scaled = self.scaler.fit_transform(self.y)
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size = 0.2)
        return X_train, X_test, y_train, y_test
