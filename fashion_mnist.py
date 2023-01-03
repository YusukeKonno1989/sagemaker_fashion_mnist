from __future__ import print_function
import os
import keras
import numpy as np
import tensorflow as tf

from keras.datasets import fashion_mnist
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.optimizers import adam_v2
from keras.callbacks import EarlyStopping

batch_size = 128
num_classes = 10
epochs = 12

# 入力画像の次元数
img_rows, img_cols = 28, 28

# データ取得
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# データをラベル化
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

labels = np.arange(len(class_names))
for label,class_name in zip(labels,class_names):
    print(label,":",class_name)

# 784次元(28×28)ベクトルに変換
X_train = X_train.reshape(X_train.shape[0], 784)
X_test = X_test.reshape(X_test.shape[0], 784)

# データの正規化
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# 学習用にラベルデータをone-hot変換
y_train = to_categorical(y_train, num_classes = len(class_names))
y_test = to_categorical(y_test, num_classes = len(class_names))

# EarlyStopping
early_stopping = EarlyStopping(
    monitor='loss',
    patience=10,
    verbose=1
)

# 最適化関数（Ir：学習率）
optimizer = adam_v2.Adam(lr=0.001)

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(256, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(128, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(len(class_names), activation=tf.nn.softmax)
])

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

hist = model.fit(
  X_train, 
  y_train, 
  epochs=50,
  validation_split=0.2,
  callbacks=[early_stopping])

model.save(os.path.join(os.environ['SM_OUTPUT_DATA_DIR'], 'model.h5'))