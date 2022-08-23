import tensorflow as tf
from tensorflow import keras
from utils import load_cifar10_datasets, set_soft_gpu

class Lenet5(keras.Model):

    def __init__(self):
        super(Lenet5, self).__init__()

        self.model = keras.Sequential([
            # conv1 [None 32 32 3] -> [None 14 14 6]
            keras.layers.Conv2D(filters=6, kernel_size=[5,5], padding="valid", strides=1),
            keras.layers.ReLU(),
            keras.layers.MaxPool2D(pool_size=[2,2], strides=2, padding="valid"),
            # conv2 [None 14 14 6] -> [None 5 5 16]
            keras.layers.Conv2D(filters=16, kernel_size=[5, 5], padding="valid", strides=1),
            keras.layers.ReLU(),
            keras.layers.MaxPool2D(pool_size=[2,2], strides=2, padding="valid"),
            # Flatten: [None 5 5 16] -> [None 5*5*16]
            keras.layers.Flatten(),
            # fc [None 5*5*16] -> [None 10]
            keras.layers.Dense(120),
            keras.layers.ReLU(),
            keras.layers.Dense(84),
            keras.layers.ReLU(),
            keras.layers.Dense(10),
        ], name="Lenet-5")

        self.loss_func = keras.losses.CategoricalCrossentropy(from_logits=True)
        self.opt = keras.optimizers.Adam(0.0001)

    def call(self, inputs, training=None, mask=None):
        return self.model.call(inputs)

def train():
    set_soft_gpu(True)
    dbtrain, dbtest = load_cifar10_datasets(128)
    model = Lenet5()
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

