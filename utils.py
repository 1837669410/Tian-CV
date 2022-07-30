import tensorflow as tf
from tensorflow import keras

def process(x, y):
    # x[0->255] -> [-1->1]
    x = tf.cast(x / 255 * 2 - 1, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.int32)
    return x, y

def load_mnist_datasets(batch_size=64):
    # [None 28 28] [None,]
    (xtrain, ytrain), (xtest, ytest) = keras.datasets.mnist.load_data()
    print(xtrain.shape, ytrain.shape, xtest.shape, ytest.shape)

    dbtrain = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
    dbtrain = dbtrain.map(process).shuffle(10000).batch(batch_size)
    dbtest = tf.data.Dataset.from_tensor_slices((xtest, ytest))
    dbtest = dbtest.map(process).shuffle(10000).batch(batch_size)

    print(next(iter(dbtrain))[0].shape, next(iter(dbtrain))[1].shape)

    return dbtrain, dbtest

