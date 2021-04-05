import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((60000, 28, 28, 1))
x_test = x_test.reshape((10000, 28, 28, 1))
# homework~ (normalization)
x_train, x_test = x_train / 255.0, x_test / 255.0
#
model = models.Sequential()
# homework ~ (CNN model)
model.add(layers.Conv2D(32, (3, 3), activation='sigmoid', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='sigmoid'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='sigmoid'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='sigmoid'))
model.add(layers.Dense(10, activation='softmax'))
# homework
model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test,  y_test, verbose=2)

test_num = 10
for i in range(test_num):
    answer = model.predict(np.expand_dims(x_test[i,:,:],axis=0))
    print('prediction : ', answer)
    print('prediction : ', np.argmax(answer))
    print('answer : ', y_test[i])

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test[i,:,:])

    plt.subplot(1,2,2)
    plt.stem(range(10), answer[0,:])
    plt.xticks([0,1,2,3,4,5,6,7,8,9])
    plt.ylim([0, 1])

    plt.show()
    # plt.show(block=False)
    # plt.pause(1)
    # plt.close("all")