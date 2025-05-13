# 导入必要的库/Импортируйте необходимые библиотеки
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from scipy.signal import spectrogram #生成STFT谱图/Генерирование спектров STFT
import pywt #小波变换生成尺度图/Вейвлет-преобразование генерирует карту масштаба
import cv2 #图像缩放/Масштабирование изображения
from sklearn.metrics import cohen_kappa_score, confusion_matrix #评估指标/Оценка показателей
from keras.models import Sequential
from keras.layers import (Conv2D, MaxPool2D, Flatten, Dense, Dropout,
                          TimeDistributed, LSTM)
from keras.optimizers import Adam
from keras import backend as K #构建深度学习模型/Построение моделей глубокого обучения

# download dataset/下载数据集/скачать набор данных
x_train = pd.read_csv("https://github.com/TAUforPython/BioMedAI/blob/main/test_datasets/MI-EEG-B9T.csv?raw=true",
                      header=None)
x_test = pd.read_csv("https://github.com/TAUforPython/BioMedAI/blob/main/test_datasets/MI-EEG-B9E.csv?raw=true",
                     header=None)
y_train = pd.read_csv("https://github.com/TAUforPython/BioMedAI/blob/main/test_datasets/2class_MI_EEG_train_9.csv?raw=true",
                      header=None)
y_test = pd.read_csv("https://github.com/TAUforPython/BioMedAI/blob/main/test_datasets/2class_MI_EEG_test_9.csv?raw=true",
                     header=None)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

#输出训练/测试样本数和类别数（二分类）
#Выходное количество обучающих/тестовых образцов и количество категорий (бинарная классификация)
n_samples_train = len(y_train)
n_samples_test = len(y_test)

print("n_samples_train:", n_samples_train)
print("n_samples_test :", n_samples_test)

# count classes/统计类/статистический класс
n_classes = len(np.unique(y_test))

print("n_classes:", n_classes)

# calculate STFT/计算STFT/Рассчитать STFT

def spectrogram_vertical(data, fs, alto, ancho, n_canales, pts_sig,
                                 pts_superpuestos):
  #fs = fs #frecuencia de muestreo
  datesets = np.zeros((data.shape[0],alto, ancho))

  # crear matriz 2D donde se guardara cada imagen del STFT/用STFT方法建立了二维矩阵图像模型
  # С помощью метода STFT была разработана двумерная матричная модель изображения
  temporal = np.zeros((alto, ancho))

  for i in range(data.shape[0]): # n muestras/n样品检测/n Выборочное тестирование
    for j in range(n_canales): # n canales/n 渠道/n оросительная канава

      sig = data.iloc[i, j*pts_sig:(j+1)*pts_sig]

      f, t, Sxx = spectrogram(sig, fs=fs, window='hann', nperseg=fs,
                              noverlap=pts_superpuestos, nfft=fs*2,
                              scaling='spectrum')

      # concatenacion vertical chanels/垂直通道连接/Вертикальное соединение каналов
      temporal[j*45:(j+1)*45, :] = Sxx[16:61, :]

    datesets[i] = temporal
    if i % 100 == 0:
      print(i)
  return datesets

# calculate scalogram CWT/计算尺度图CWT/Расчетная масштабная диаграмма CWT

def scalogram_vertical(data, fs, alto, ancho, n_canales, pts_sig):
  dim = (int(np.floor(ancho/2)), int(np.floor(alto/2))) # ancho, alto

  # Wavelet Morlet 3-3/小波变换/вейвлет-преобразование (математика)
  # frequency 8 - 30 Hz/频率/частота
  scales = pywt.scale2frequency('cmor3-3', np.arange(8,30.5,0.5)) / (1/fs)

  datesets = np.zeros((data.shape[0], int(np.floor(alto/2)),
                    int(np.floor(ancho/2))))

  temporal = np.zeros((alto, ancho))

  for i in range(data.shape[0]):
    for j in range(n_canales):

      sig = data.iloc[i, j*pts_sig:(j+1)*pts_sig]

      coef, freqs = pywt.cwt(sig, scales, 'cmor3-3',
                             sampling_period = (1 / fs))

      temporal[j*45:(j+1)*45, :] = abs(coef)

    resized = cv2.resize(temporal, dim, interpolation=cv2.INTER_AREA)
    datesets[i] = resized
    if i % 100 == 0:
      print(i)
  return datesets

initial = time.time()

# STFT
x_train = spectrogram_vertical(x_train, 250, 135, 31, 3, 1000, 225)
x_test = spectrogram_vertical(x_test, 250, 135, 31, 3, 1000, 225)

# CWT
#x_train = scalogram_vertical(x_train, 250, 135, 1000, 3, 1000)
#x_test = scalogram_vertical(x_test, 250, 135, 1000, 3, 1000)

fin = time.time()
print("time_elapsed:", fin - initial)

print(x_train.shape)
print(x_test.shape)

x = np.ceil(np.max(x_train))

# convert to float
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= x
x_test /= x

plt.figure()
plt.imshow(x_train[1],  aspect='auto')
plt.colorbar()
plt.show()

#  reshape a 4D (for CNN-2D)
#x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
#x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

# convert  3D to 5D (CNN-2D + LSTM)
x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1], x_train.shape[2], 1))
x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1], x_test.shape[2], 1))

print(x_train.shape)
print(x_test.shape)

# crear red neuronal CNN-2D/纯CNN模型建模 (CNN_2D)/Чистое моделирование CNN-моделей (CNN_2D)

def CNN_2D(num_filter, size_filter, n_neurons):
  model = Sequential()
  model.add(Conv2D(num_filter, size_filter, activation='relu', padding='same',
                   input_shape=x_train.shape[1:]))
  model.add(MaxPool2D((2,2)))
  model.add(Conv2D(num_filter, size_filter, activation='relu', padding='same'))
  model.add(MaxPool2D((2,2)))
  model.add(Flatten())
  model.add(Dense(n_neurons, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(n_classes, activation='softmax'))

  optimizer = Adam(learning_rate=0.001)
  model.compile(optimizer = optimizer,
                loss = 'sparse_categorical_crossentropy',
                metrics = ['accuracy'])
  return model

# crear red neuronal CNN-2D + LSTM/CNN-LSTM混合模型 (CNN_2D_LSTM_TD)建模 /Гибридная модель CNN-LSTM (CNN_2D_LSTM_TD) моделирование

def CNN_2D_LSTM_TD(num_filter, size_filter, n_neurons, units_LSTM):
  model = Sequential()
  model.add(TimeDistributed(Conv2D(num_filter, size_filter, activation='relu',
                                   padding='same'),
                            input_shape=x_train.shape[1:]))
  model.add(TimeDistributed(MaxPool2D((2,2))))
  model.add(TimeDistributed(Conv2D(num_filter, size_filter, activation='relu',
                                   padding='same')))
  model.add(TimeDistributed(MaxPool2D((2,2))))
  model.add(TimeDistributed(Flatten()))
  model.add(LSTM(units_LSTM, activation='tanh', dropout=0.5))
  model.add(Dense(n_neurons, activation='relu'))
  model.add(Dense(n_classes, activation='softmax'))

  optimizer = Adam(learning_rate=1e-3)
  model.compile(optimizer = optimizer,
                loss = 'sparse_categorical_crossentropy',
                metrics = ['accuracy'])
  return model

#训练与评估/Обучение и оценка
initial = time.time()
array_loss = []
array_acc = []
array_kappa = []
for i in range(5):
  print("Iteration:", i+1)


  #model = CNN_2D(4, (3,3), 32)
  model = CNN_2D_LSTM_TD(4, (3,3), 32, 4)

 # history = model.fit(x_train, y_train, epochs=40, batch_size=36,
 #                     validation_data=(x_test, y_test), verbose=0)

  history = model.fit(x_train, y_train, epochs=70, batch_size=36,
                       validation_split = 0.1, verbose=0)

  test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

  array_loss.append(test_loss)
  print("loss: ", test_loss)
  array_acc.append(test_acc)
  print("accuracy: ", test_acc)

  # 评估指标: 准确率/Показатель оценки: Точность
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.grid()
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend(['train', 'val'])
  plt.show()
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.grid()
  plt.xlabel('Epochs')
  plt.ylabel('Cross-Entropy')
  plt.legend(['train', 'val'])
  plt.show()

  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper left')
  plt.show()

  # 评估指标: Kappa系数/Показатель оценки: коэффициент Каппа
  probabilidades = model.predict(x_test)

  y_pred = np.argmax(probabilidades, 1)

  # calculate kappa cohen
  kappa = cohen_kappa_score(y_test, y_pred)
  array_kappa.append(kappa)
  print("kappa: ", kappa)
  matriz_confusion = confusion_matrix(y_test, y_pred)
  print("confusion matrix:\n", matriz_confusion)

  # Интерпретация
  # (\kappa = 1): Полное согласие, модель идеально предсказывает все классы.
  # (\kappa = 0): Согласие на уровне случайного угадывания.
  # (\kappa < 0): Согласие хуже, чем случайное угадывание.

  # 评估指标: 混淆矩阵/Показатель оценки: матрица путаницы
  from sklearn.metrics import ConfusionMatrixDisplay

  labels = ["Left MI", "Right MI"]

  disp = ConfusionMatrixDisplay(confusion_matrix=matriz_confusion, display_labels=labels)

  disp.plot(cmap=plt.cm.Blues)
  plt.show()

  # 绘制训练/验证的准确率和损失曲线/Постройте кривые точности и потерь для обучения и проверки.
  print()
  print("Resultados:")
  print("loss:", array_loss)
  print("accuracy:", array_acc)
  print("kappa:", array_kappa)
  fin = time.time()
  time_elapsed = fin - initial
  print("time_elapsed:", time_elapsed)

  # 输出模型结构/Структура выходной модели
  model.summary()

  # 统计指标：均值，标准差，最大值，总耗时
  # среднее значение, стандартное отклонение, максимальное значение, общее затраченное время
  print("Mean Accuracy: %.4f" % np.mean(array_acc))
  print("std: (+/- %.4f)" % np.std(array_acc))
  print("Mean Kappa: %.4f" % np.mean(array_kappa))
  print("std: (+/- %.4f)" % np.std(array_kappa))
  print("Max Accuracy: %.4f" % np.max(array_acc))
  print("Max Kappa: %.4f" % np.max(array_kappa))
  print("time_elapsed:", int(time_elapsed))