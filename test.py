# Created by jikangwang at 3/17/19


from __future__ import absolute_import, division, print_function

from tensorflow import keras


fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_images = train_images / 255.0

test_images = test_images / 255.0


model = keras.models.load_model('model.h5')

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)