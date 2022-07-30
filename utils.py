import tensorflow as tf
from tensorflow import keras

def process(x, y):
    # x[0->255] -> [-1->1]
    x = tf.cast(x / 255 * 2 - 1, dtype=tf.float32)
    y = tf.squeeze(y)
    y = tf.cast(y, dtype=tf.int32)
    return x, y

def load_cifar10_datasets(batch_size=64):
    # [None 28 28] [None,]
    (xtrain, ytrain), (xtest, ytest) = keras.datasets.cifar10.load_data()
    print(xtrain.shape, ytrain.shape, xtest.shape, ytest.shape)

    dbtrain = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
    dbtrain = dbtrain.map(process).shuffle(10000).batch(batch_size)
    dbtest = tf.data.Dataset.from_tensor_slices((xtest, ytest))
    dbtest = dbtest.map(process).shuffle(10000).batch(batch_size)

    print(next(iter(dbtrain))[0].shape, next(iter(dbtrain))[1].shape)

    return dbtrain, dbtest

def set_soft_gpu(soft_gpu):
    import tensorflow as tf
    if soft_gpu:
        # 返回电脑上可用的GPU列表
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

