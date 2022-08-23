import tensorflow as tf
from tensorflow import keras
from utils import set_soft_gpu, load_cifar10_datasets

class InceptionV1(keras.layers.Layer):

    def __init__(self, filters):
        super(InceptionV1, self).__init__()
        self.p1_1 = keras.layers.Conv2D(filters=filters[0], kernel_size=[1,1], strides=1, padding="same", activation="relu")
        self.p2_1 = keras.layers.Conv2D(filters=filters[1], kernel_size=[1,1], strides=1, padding="same", activation="relu")
        self.p2_3 = keras.layers.Conv2D(filters=filters[2], kernel_size=[3,3], strides=1, padding="same", activation="relu")
        self.p3_1 = keras.layers.Conv2D(filters=filters[3], kernel_size=[1,1], strides=1, padding="same", activation="relu")
        self.p3_5 = keras.layers.Conv2D(filters=filters[4], kernel_size=[5,5], strides=1, padding="same", activation="relu")
        self.p4_M = keras.layers.MaxPool2D(pool_size=[3,3], strides=1, padding="same")
        self.p4_1 = keras.layers.Conv2D(filters=filters[5], kernel_size=[1,1], strides=1, padding="same", activation="relu")

    def call(self, inputs, **kwargs):
        out_1 = self.p1_1(inputs)
        out_2 = self.p2_3(self.p2_1(inputs))
        out_3 = self.p3_5(self.p3_1(inputs))
        out_4 = self.p4_1(self.p4_M(inputs))
        out = tf.concat((out_1, out_2, out_3, out_4), axis=3)
        return out

class GoogLenetV1(keras.Model):

    def __init__(self):
        super(GoogLenetV1, self).__init__()

        self.stage1 = keras.Sequential([
            # [None 32 32 3] -> [None 32 32 64] -> [None 16 16 64]
            keras.layers.Conv2D(filters=64, kernel_size=[7,7], strides=1, padding="same", activation="relu"),
            keras.layers.MaxPool2D(pool_size=[3,3], strides=2, padding="same"),
        ])
        self.stage2 = keras.Sequential([
            # [None 16 16 64] -> [None 16 16 192] -> [None 16 16 192] -> [None 8 8 192]
            keras.layers.Conv2D(filters=192, kernel_size=[1,1], strides=1, padding="same", activation="relu"),
            keras.layers.Conv2D(filters=192, kernel_size=[3,3], strides=1, padding="same", activation="relu"),
            keras.layers.MaxPool2D(pool_size=[3,3], strides=2, padding="same"),
        ])
        self.stage3 = keras.Sequential([
            # [None 8 8 192] -> [None 8 8 256] -> [None 8 8 480] -> [None 4 4 480]
            InceptionV1([64, 96, 128, 16, 32, 32]),
            InceptionV1([128, 128, 192, 32, 96, 64]),
            keras.layers.MaxPool2D(pool_size=[3,3], strides=2, padding="same"),
        ])
        self.stage4 = keras.Sequential([
            # [None 4 4 480] -> [None 4 4 512] -> [None 4 4 512] -> [None 4 4 512] -> [None 4 4 528] -> [None 4 4 832] -> [None 2 2 832]
            InceptionV1([192, 96, 208, 16, 48, 64]),
            InceptionV1([160, 112, 224, 24, 64, 64]),
            InceptionV1([128, 128, 256, 24, 64, 64]),
            InceptionV1([112, 144, 288, 32, 64, 64]),
            InceptionV1([256, 160, 320, 32, 128, 128]),
            keras.layers.MaxPool2D(pool_size=[3,3], strides=2, padding="same"),
        ])
        self.stage5 = keras.Sequential([
            # [None 2 2 832] -> [None 2 2 832] -> [None 2 2 1024] -> [None 1 1 1024] -> [None 1024]
            InceptionV1([256, 160, 320, 32, 128, 128]),
            InceptionV1([384, 192, 384, 48, 128, 128]),
            keras.layers.GlobalAvgPool2D(),
            keras.layers.Flatten(),
        ])
        self.dropout = keras.layers.Dropout(0.4)
        self.dense1 = keras.layers.Dense(10)

        self.loss_func = keras.losses.CategoricalCrossentropy(from_logits=True)
        self.opt = keras.optimizers.Adam(0.0002)

    def call(self, inputs, training=None, mask=None):
        out = self.stage1(inputs)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.stage5(out)
        out = self.dropout(out)
        out = self.dense1(out)
        return out

def train():
    set_soft_gpu(True)
    dbtrain, dbtest = load_cifar10_datasets(128)
    model = GoogLenetV1()
    model.build(input_shape=[None, 32, 32, 3])
    model.summary()
    for e in range(10):
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

