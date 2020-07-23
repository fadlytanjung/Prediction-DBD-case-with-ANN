import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD, Adagrad, RMSprop, Adadelta, Adamax, Adam
from tensorflow.keras.models import model_from_json
from tensorflow.keras import backend as K
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import math

class Preprocessing:
    def __init__(self):
        self.trainScore = 0
        self.testScore = 0
        self.rmseTrain = 0
        self.rmseTest = 0

    "load data input"
    def load_data(self, file):
        data = pd.read_excel(file)

        data = data.sort_values([4,1, 0], ascending=[True, True, True])

        return data.values

    '''milih baris data yang obat'''
    def selecting_data(self, data):
        temp = np.zeros(4, )
        for i in data:
            if i[3] == 1:
                pecah_date = i[0].split('-')

                nama_obat = i[1].upper()

                np.put(i, [0], (pecah_date[1]))
                np.put(i, [1], nama_obat)
                np.put(i,[3],'OBAT')
                temp = np.vstack((temp, i))
        temp = np.delete(temp, (0, 0), 0)
        return temp

    '''hitung jumlah obat perbulan'''
    def sum_data(self, data):
        df = pd.DataFrame(data)
        df = df.groupby([0, 1])
        new_data = np.zeros(4, )

        j = 1
        for name, group in df:

            j += 1
            stok = 0
            for i in group.values:
                stok += i[2]

                isi = i
            np.put(isi, 2, stok)
            new_data = np.vstack((new_data, isi))
        new_data = np.delete(new_data, 0, axis=-0)

        return new_data

    def simpan_file(self, data):
        np.savetxt("new_dataset.csv", data, delimiter=',', fmt='%s')

    def one_hot(self, data):
        values = data[:, 4]
        # define example
        '''encode nama obat ke integer'''
        # integer encode
        self.label_encoder = LabelEncoder()
        integer_encoded = self.label_encoder.fit_transform(values)
        data[:, 4] = integer_encoded
        return data

    '''hapus kolom yg gak dipake'''
    def hapus_kolom(self, data, kolom):
        return np.delete(data, kolom, axis=1)

    '''bagi train sama test data'''
    def split_data(self, dataset):
        train_size = int(len(dataset) * 0.80)
        test_size = len(dataset) - train_size
        train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
        return train, test

    def normalisasi(self, data):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = self.scaler.fit_transform(np.reshape(data[:,1],(-1,1)))
        data[:,1] = np.reshape(dataset,(1,-1))
        return data

    def create_dataset(self, dataset, look_back=1):
        dataX, dataY = [], []
        hapus_baris = []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i, :]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 1])

            if i>0 and dataset[i,0] != dataset[i-1,0]:
                hapus_baris.append((i-1))
        dataX = np.array(dataX)
        dataY = np.array(dataY)
        dataX = np.delete(dataX,hapus_baris,axis=0)
        dataY = np.delete(dataY,hapus_baris,axis=0)

        return dataX, dataY


class ANN:
    def __init__(self):
        self.model_loaded = False

    def set_param(self, neuron=4, optimizer="Adam", epoch=10, batch_size=1, lr=0.001, activation='relu'):
        self.neuron = neuron
        self.optimizer = optimizer
        self.epoch = epoch
        self.batch_size = batch_size
        self.model_loaded = False
        self.learning_rate = lr
        self.activation = activation

    def soft_acc(self, y_true, y_pred):
        return K.mean(K.equal(K.round(y_true), K.round(y_pred)))

    '''pembuatan model'''
    def training(self, trainX, trainY, testX, testY, preprocessing):
        model = Sequential()
        model.add(Dense(2, input_dim=2, activation=self.activation))  # inputlayer
        model.add(Dense(self.neuron, activation=self.activation))  # hiddenlayer
        model.add(Dense(1, activation='linear'))  # outputlayer
        if 'SGD' in self.optimizer:
            opt = SGD(lr=self.learning_rate, momentum=0.0, decay=0.0, nesterov=False)

        if 'RMSProp' in self.optimizer:
            opt = RMSprop(lr=self.learning_rate, rho=0.9, epsilon=None, decay=0.0)

        if 'Adgrad' in self.optimizer:
            opt = Adagrad(lr=self.learning_rate, epsilon=None, decay=0.0)

        if 'Adamax' in self.optimizer:
            opt = Adamax(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)

        if 'Adam' in self.optimizer:
            opt = Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

        if 'Adadelta' in self.optimizer:
            opt = Adadelta(lr=self.learning_rate, rho=0.95, epsilon=None, decay=0.0)

        model.compile(loss='mean_squared_error', optimizer=opt)
        self.history = model.fit(trainX, trainY, epochs=self.epoch, batch_size=self.batch_size, verbose=2, validation_data=(testX,testY))

        # save history
        loss_history = self.history.history["loss"]
        # acc_history = self.history.history["soft_acc"]
        testing_loss_history = self.history.history["val_loss"]
        # testing_acc_history = self.history.history["val_soft_acc"]
        loss = np.array(loss_history)
        np.savetxt("static/loss_history.txt", loss, delimiter=",")
        # acc = np.array(acc_history)
        # np.savetxt("static/acc_history.txt", acc, delimiter=",")
        tes_loss = np.array(testing_loss_history)
        np.savetxt("static/testing_loss_history.txt", tes_loss, delimiter=",")
        # tes_acc = np.array(testing_acc_history)
        # np.savetxt("static/testing_acc_history.txt", tes_acc, delimiter=",")

        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)

        model.save_weights('weights.h5')


        testPredict = model.predict(testX)
        testPredict = preprocessing.scaler.inverse_transform(testPredict)

        # Estimate model performance
        trainScore = model.evaluate(trainX, trainY, verbose=0)
        print('Train Score: %.5f MSE (%.5f RMSE)' % (trainScore, math.sqrt(trainScore)))
        testScore = model.evaluate(testX, testY, verbose=0)
        print('Test Score: %.5f MSE (%.5f RMSE)' % (testScore, math.sqrt(testScore)))
        self.trainScore = trainScore
        self.testScore = testScore
        self.rmseTrain = math.sqrt(trainScore)
        self.rmseTest = math.sqrt(testScore)
        score = np.array([self.trainScore,self.testScore,self.rmseTrain,self.rmseTest]);
        np.savetxt("static/score.txt",score, delimiter=";")


        # plot baseline and predictions
        testY = preprocessing.scaler.inverse_transform(np.reshape(testY,(-1,1)))
        testX = testX.astype(int)
        testY = testY.astype(int)
        testPredict = testPredict.astype(int)
        obat_asli = preprocessing.label_encoder.inverse_transform(testX[:,0])
        testX = testX.astype("S100")
        testY = testY.astype("S100")
        testX[:, 0] = obat_asli
        simpan = np.hstack((testX,testY))
        simpan = np.hstack((simpan,testPredict))
        simpan = np.delete(simpan,1,axis=1)
        df = pd.DataFrame(simpan)
        df.columns = ["Obat","Actual","Predicted"]

        writer = pd.ExcelWriter('static/hasil_training.xlsx', engine='xlsxwriter')
        df.to_excel(writer, "Sheet1")
        writer.save()
        K.clear_session()

    def load_model(self):
        # load json file
        json_file = open("model.json", "r")
        loaded_model_json = json_file.read()
        json_file.close()

        # load weight
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights("weights.h5")

        self.model.compile(loss='mean_squared_error', optimizer=self.optimizer, metrics=[self.soft_acc])
        self.model_loaded = True

    def prediction(self, datax, bulan, start_bulan, start_tahun, preprocessing):
        if self.model_loaded == False:
            print("Model loaded")
            self.load_model()


        tulis = np.zeros((4,), dtype="S250")
        '''prediksi sebanyak variabel bulan'''
        for bln in range(bulan):
            # make predictions
            '''masukkan hasil prediksi ke feature untuk ditabelin'''
            predict = self.model.predict(datax)
            predict = preprocessing.scaler.inverse_transform(np.reshape(predict,(-1,1)))
            new_data = np.zeros(4, )
            x = 0
            for i in datax:
                temp = np.array((start_bulan,start_tahun,i[0]))
                new_data = np.vstack((new_data, np.append(temp, predict[x])))
                x += 1
            new_data = np.delete(new_data, 0, axis=0)

            result_test = new_data  # preprocessing.scaler.inverse_transform(new_data)
            result_test = np.rint(result_test)
            obat = result_test[:, 2]
            obat = obat.astype(int)
            obat_asli = preprocessing.label_encoder.inverse_transform(obat)

            start_bulan += 1
            if start_bulan>12:
                start_tahun+=1
                start_bulan=1

            tampilkan = result_test
            tampilkan = tampilkan.astype("S250")
            tampilkan[:, 2] = obat_asli
            # tampilkan[:,0] = start_bulan
            df = pd.DataFrame(tampilkan)
            tulis = np.vstack((tulis, tampilkan))
            print("yang ke-", bln)
            df.columns = ['Bulan ke','Tahun','Obat', 'Stok']
            # print(df)

            # result_test[:, 0] = start_bulan
            # result_test[:,1] = start_tahun
            datax = result_test[:,2:4]
            #print("bulan")
            #print(datax)
            datax = preprocessing.normalisasi(datax)

        tulis = np.delete(tulis, 0, axis=0)
        df = pd.DataFrame(tulis)
        df.columns = ['Bulan ke','Tahun','Obat', 'Stok']

        writer = pd.ExcelWriter('static/prediksi.xlsx', engine='xlsxwriter')
        df.to_excel(writer, "Sheet1")
        writer.save()
        K.clear_session()

    def save_image(self):
        data = pd.read_csv("static/loss_history.txt")
        data = data.values
        data2= pd.read_csv("static/testing_loss_history.txt")
        data2 = data2.values
        # summarize history for loss
        plt.plot(data)
        plt.plot(data2)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train','test'], loc='upper left')
        plt.savefig('static/loss.png')
        plt.close()

        data = pd.read_csv("static/testing_loss_history.txt")
        data = data.values
        # summarize history for loss
        plt.plot(data)
        plt.title('model test loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['test'], loc='upper left')
        plt.savefig('static/loss_test.png')
        plt.close()


if __name__ == "__main__":
    # fix random seed for reproducibility#
    np.random.seed(7)
    preprocessing = Preprocessing()

    data = preprocessing.load_data("data/datatraining.xlsx")

    #data = preprocessing.selecting_data(data)
    #data = preprocessing.sum_data(data)
    data = preprocessing.one_hot(data)

  
    #data = data[data[:, 4].argsort()]
    data = preprocessing.hapus_kolom(data, [0, 1, 2, 3, 5, 7])
    data = data.astype('float32')
    
    normal_data = preprocessing.normalisasi(data)
    train, test = preprocessing.split_data(normal_data)

    '''buat label'''
    look_back = 1
    trainx, trainY = preprocessing.create_dataset(train, look_back)
    testx, testY = preprocessing.create_dataset(test, look_back)

    # trainY = trainY.reshape(len(trainY),1)
    # testY = testY.reshape(len(testY),1)p

    nn = ANN()
    nn.set_param(neuron=8, optimizer="Adam", epoch=5, batch_size=16)
    nn.training(trainx, trainY, testx, testY, preprocessing)
    nn.prediction(trainx, 12, 1, 2020, preprocessing)

