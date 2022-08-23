import tensorflow as tf
from tensorflow import keras
from utils import set_soft_gpu, load_cifar10_datasets

class Alexnet(keras.Model):

    def __init__(self):
        super(Alexnet, self).__init__()

        self.model = keras.Sequential([
            # conv1 [None 32 32 3] -> [None 8 8 48] -> [None 8 8 48]
            keras.layers.Conv2D(filters=48, kernel_size=[11,11], strides=4, padding="same"),
            keras.layers.ReLU(),
            # conv2 [None 8 8 48] -> [None 8 8 128] -> [None 4 4 128]
            keras.layers.Conv2D(filters=128, kernel_size=[5,5], strides=1, padding="same"),
            keras.layers.MaxPool2D(pool_size=[2,2], strides=2, padding="same"),
            keras.layers.ReLU(),
            # conv3 [None 4 4 128] -> [None 4 4 192] -> [None 2 2 192]
            keras.layers.Conv2D(filters=192, kernel_size=[3,3], strides=1, padding="same"),
            keras.layers.MaxPool2D(pool_size=[2,2], strides=2, padding="same"),
            keras.layers.ReLU(),
            # conv4 [None 2 2 192] -> [None 2 2 192]
            keras.layers.Conv2D(filters=192, kernel_size=[3,3], strides=1, padding="same"),
            keras.layers.ReLU(),
            # conv5 [None 2 2 192] -> [None 2 2 192] -> [None 1 1 192]
            keras.layers.Conv2D(filters=192, kernel_size=[3,3], strides=1, padding="same"),
            keras.layers.MaxPool2D(pool_size=[2,2], strides=2, padding="same"),
            keras.layers.ReLU(),
            # Flatten [None 1 1 192] -> [None 192]
            keras.layers.Flatten(),
            # [None 192] -> [None 100] -> [None 64] -> [None 10]
            keras.layers.Dense(100),
            keras.layers.ReLU(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(64),
            keras.layers.ReLU(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(10),
        ], name="Alexnet")

        self.loss_func = keras.losses.CategoricalCrossentropy(from_logits=True)
        self.opt = keras.optimizers.Adam(0.0001)

    def call(self, inputs, training=None, mask=None):
        return self.model.call(inputs)

def train():
    set_soft_gpu(True)
    dbtrain, dbtest = load_cifar10_datasets(128)
    model = Alexnet()
    model.build(input_shape=[None, 32, 32, 3])
    model.summary()
    for e in range(5):
        for step, (x, y) in enumerate(dbtrain):
            with tf.GradientTape() as tape:
                logtis = model.call(x)
                y = tf.one_hot(y, depth=10)
                loss = model.loss_func(y, logtis)
                grads = tape.gradient(loss, model.trainable_variables)
            model.opt.apply_gradients(zip(grads, model.trainable_variables))
            if step % 100 == 0:
                print("epoch:{} | step:{} | loss:{}".format(e, step, loss))

        total_num = 0
        total_acc = 0

        for step, (x, y) in enumerate(dbtest):
            total_num += x.shape[0]
            # [None 10] -> [None 1]
            out = model.call(x)
            pred = tf.nn.softmax(out, axis=1)
            pred = tf.cast(tf.argmax(pred, axis=1), dtype=tf.int32)
            pred = tf.cast(tf.equal(pred, y), dtype=tf.int32)
            total_acc += tf.reduce_sum(pred)
        print("epoch:{} | acc:{}".format(e, total_acc / total_num))

if __name__ == "__main__":
    train()