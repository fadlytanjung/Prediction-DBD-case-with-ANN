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
    def training(self, X_train, X_test, y_train, y_test, preprocessing):
        model = Sequential()
        model.add(Dense(4, input_dim=8, activation=self.activation))  # inputlayer
        model.add(Dense(self.neuron, activation=self.activation))  # hiddenlayer
        model.add(Dense(1, activation='linear'))  # outputlayer
        if 'SGD' in self.optimizer:
            opt = SGD(lr=0.001)
            
        if 'RMSProp' in self.optimizer:
            opt = RMSprop(lr=0.001)

        if 'Adgrad' in self.optimizer:
            opt = Adgrad(lr=0.001)
            
        if 'Adamax' in self.optimizer:
            opt = Adamax(lr=0.001)

        if 'Adam' in self.optimizer:
            opt = Adam(lr=0.001)

        if 'Adadelta' in self.optimizer:
            opt = Adadelta(lr=0.001)
            
        model.compile(loss='mean_squared_error', optimizer=opt)
        self.history = model.fit(X_train, y_train, epochs=self.epoch, batch_size=self.batch_size, verbose=2, validation_data=(X_test,y_test))

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
        
        testPredict = model.predict(X_test)
        testPredict = preprocessing.scaler.inverse_transform(testPredict)
        
        # Estimate model performance
        trainScore = model.evaluate(X_train, y_train, verbose=0)
        print('Train Score: %.5f MSE (%.5f RMSE)' % (trainScore, math.sqrt(trainScore)))
        testScore = model.evaluate(X_test, y_test, verbose=0)
        print('Test Score: %.5f MSE (%.5f RMSE)' % (testScore, math.sqrt(testScore)))
        self.trainScore = trainScore
        self.testScore = testScore
        self.rmseTrain = math.sqrt(trainScore)
        self.rmseTest = math.sqrt(testScore)
        score = np.array([self.trainScore,self.testScore,self.rmseTrain,self.rmseTest]);
        np.savetxt("static/score.txt",score, delimiter=";")
        
        # plot baseline and predictions
#         X_test = preprocessing.scaler.inverse_transform(X_test[:,0])
        y_pred = model.predict(X_test)
        y_predict_sample_orig = preprocessing.scaler.inverse_transform(y_pred)
        y_test = preprocessing.scaler.inverse_transform(np.reshape(y_test,(-1,1)))
        kecamatan_asli = preprocessing.label_encoder.fit_transform(X_test[:,0])
        df = pd.DataFrame({'Kecamatan': kecamatan_asli.flatten(),'Aktual': y_test.flatten(), 'Prediksi': y_predict_sample_orig.flatten()})
        writer = pd.ExcelWriter('static/hasil_training.xlsx', engine='xlsxwriter')
        df.to_excel(writer, "Sheet1")
        writer.save()
        K.clear_session()
        
    def load_model(self, path):
        # load json file
        json_file = open("model.json", "r")
        loaded_model_json = json_file.read()
        json_file.close()

        # load weight
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights(path)
        self.model.summary()
        self.model.compile(loss='mean_squared_error', optimizer=self.optimizer, metrics=[self.soft_acc])
        self.model_loaded = True

        return 'Model Loaded'
        
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
            kecamatan = result_test[:, 0]
            kecamatan = kecamatan.astype(int)
            kecamatan_asli = preprocessing.label_encoder.inverse_transform(kecamatan)
            start_bulan += 1
            if start_bulan>12:
                start_tahun+=1
                start_bulan=1

            tampilkan = result_test
            tampilkan = tampilkan.astype("S250")
            tampilkan[:, 0] = kecamatan_asli
#             # tampilkan[:,0] = start_bulan
#             df = pd.DataFrame({'Bulan ke': result_test[:, 0].flatten(),'Tahun': result_test[:,1].flatten(),'Kecamatan': kecamatan_asli.flatten(),'Jumlah Kasus': predict.flatten()})
            df = pd.DataFrame(tampilkan)
            tulis = np.vstack((tulis, tampilkan))
            print("yang ke-", bln)
            print(tulis)
            #df.columns = ['Bulan ke','Tahun','Kecamatan', 'Jumlah Kasus']
            # result_test[:, 0] = start_bulan
            # result_test[:,1] = start_tahun
            
            #datax = result_test[:,2:4]
            #datax = preprocessing.normalisasi(datax)
        #exit()
        tulis = np.delete(tulis, 0, axis=0)
        df = pd.DataFrame(tulis)
        df.columns = ['Bulan ke','Tahun','Kecamatan', 'Jumlah Kasus']
        print(df)
        #writer = pd.ExcelWriter('static/prediksi.xlsx', engine='xlsxwriter')
        #df.to_excel(writer, "Sheet1")
        #writer.save()
        K.clear_session()