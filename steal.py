# Created by jikangwang at 3/17/19

from tensorflow import keras
import tensorflow as tf
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_images = train_images / 255.0

test_images = test_images / 255.0

# We need to create a new model which have the same behaviors as attack_model
target_model = keras.models.load_model('model.h5')

steal_model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

target_confidential=target_model.predict(test_images)

steal_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

steal_model.fit(test_images,target_confidential.argmax(axis=1), epochs=5)

test_loss, test_acc = steal_model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

