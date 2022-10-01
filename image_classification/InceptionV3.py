# https://www.jianshu.com/p/d59db7513fd2 架构图来源

import tensorflow as tf
from tensorflow import keras
from InceptionV2 import InceptionV2
from utils import load_cifar10_datasets, set_soft_gpu

class InceptionV3_1(keras.layers.Layer):

    def __init__(self, filter):
        super(InceptionV3_1, self).__init__()
        # 第一条路 [1,1]
        self.path1 = keras.Sequential([
            keras.layers.Conv2D(filters=filter[0], kernel_size=[1,1], strides=1, padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
        ])
        # 第二条路 [1,1] [1,7] [7,1]
        self.path2 = keras.Sequential([
            keras.layers.Conv2D(filters=filter[1], kernel_size=[1,1], strides=1, padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Conv2D(filters=filter[1], kernel_size=[1,7], strides=1, padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Conv2D(filters=filter[0], kernel_size=[7,1], strides=1, padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
        ])
        # 第三条路 [1,1] [7,1] [1,7] [7,1] [1,7]
        self.path3 = keras.Sequential([
            keras.layers.Conv2D(filters=filter[1], kernel_size=[1,1], strides=1, padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Conv2D(filters=filter[1], kernel_size=[7,1], strides=1, padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Conv2D(filters=filter[1], kernel_size=[1,7], strides=1, padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Conv2D(filters=filter[1], kernel_size=[7,1], strides=1, padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Conv2D(filters=filter[0], kernel_size=[1,7], strides=1, padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
        ])
        # 第四条路 [pool 3,3] [1,1]
        self.path4 = keras.Sequential([
            keras.layers.MaxPool2D(pool_size=[3,3], strides=1, padding="same"),
            keras.layers.Conv2D(filters=filter[0], kernel_size=[1,1], strides=1, padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
        ])

    def call(self, inputs, **kwargs):
        # 第一条路
        out1 = self.path1(inputs)
        out2 = self.path2(inputs)
        out3 = self.path3(inputs)
        out4 = self.path4(inputs)
        out = tf.concat((out1, out2, out3, out4), axis=3)
        return out

class InceptionV3_2(keras.layers.Layer):

    def __init__(self):
        super(InceptionV3_2, self).__init__()

        self.path1 = keras.Sequential([
            keras.layers.Conv2D(filters=320, kernel_size=[1,1], strides=1, padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
        ])

        self.path2_1 = keras.layers.Conv2D(filters=384, kernel_size=[1,1], strides=1, padding="same", activation="relu")
        self.bn2_1 = keras.layers.BatchNormalization()
        self.path2_2_1 = keras.layers.Conv2D(filters=384, kernel_size=[1,3], strides=1, padding="same", activation="relu")
        self.bn2_2_1 = keras.layers.BatchNormalization()
        self.path2_2_2 = keras.layers.Conv2D(filters=384, kernel_size=[3,1], strides=1, padding="same", activation="relu")
        self.bn2_2_2 = keras.layers.BatchNormalization()

        self.path3_1 = keras.layers.Conv2D(filters=448, kernel_size=[1,1], strides=1, padding="same", activation="relu")
        self.bn3_1 = keras.layers.BatchNormalization()
        self.path3_2 = keras.layers.Conv2D(filters=384, kernel_size=[3,3], strides=1, padding="same", activation="relu")
        self.bn3_2 = keras.layers.BatchNormalization()
        self.path3_3_1 = keras.layers.Conv2D(filters=384, kernel_size=[1,3], strides=1, padding="same", activation="relu")
        self.bn3_3_1 = keras.layers.BatchNormalization()
        self.path3_3_2 = keras.layers.Conv2D(filters=384, kernel_size=[3,1], strides=1, padding="same", activation="relu")
        self.bn3_3_2 = keras.layers.BatchNormalization()

        self.path4 = keras.Sequential([
            keras.layers.MaxPool2D(pool_size=[3,3], strides=1, padding="same"),
            keras.layers.Conv2D(filters=192, kernel_size=[1,1], strides=1, padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU()
        ])

    def call(self, inputs, **kwargs):
        out1 = self.path1(inputs)

        out2_1 = self.bn2_1(self.path2_1(inputs))
        out2_2_1 = self.bn2_2_1(self.path2_2_1(out2_1))
        out2_2_2 = self.bn2_2_2(self.path2_2_2(out2_1))
        out2 = tf.concat((out2_2_1, out2_2_2), axis=3)

        out3_1 = self.bn3_1(self.path3_1(inputs))
        out3_2 = self.bn3_2(self.path3_2(out3_1))
        out3_3_1 = self.bn3_3_1(self.path3_3_1(out3_2))
        out3_3_2 = self.bn3_3_2(self.path3_3_2(out3_2))
        out3 = tf.concat((out3_3_1, out3_3_2), axis=3)

        out4 = self.path4(inputs)

        out = tf.concat((out1, out2, out3, out4), axis=3)
        return out

class GoogLenetV3(keras.Model):

    def __init__(self):
        super(GoogLenetV3, self).__init__()

        # stage3的前置
        self.conv3_1 = keras.layers.Conv2D(filters=384, kernel_size=[3,3], strides=2, padding="same", activation="relu")
        self.bn3_1 = keras.layers.BatchNormalization()

        self.conv3_2_1 = keras.layers.Conv2D(filters=64, kernel_size=[1,1], strides=1, padding="same", activation="relu")
        self.bn3_2_1 = keras.layers.BatchNormalization()
        self.conv3_2_2 = keras.layers.Conv2D(filters=96, kernel_size=[3,3], strides=1, padding="same", activation="relu")
        self.bn3_2_2 = keras.layers.BatchNormalization()
        self.conv3_2_3 = keras.layers.Conv2D(filters=96, kernel_size=[3,3], strides=2, padding="same", activation="relu")
        self.bn3_2_3 = keras.layers.BatchNormalization()

        self.maxpool3_1 = keras.layers.MaxPool2D(pool_size=[3,3], strides=2, padding="same")

        # # stage4的前置
        self.conv4_1_1 = keras.layers.Conv2D(filters=192, kernel_size=[1,1], strides=1, padding="same", activation="relu")
        self.bn4_1_1 = keras.layers.BatchNormalization()
        self.conv4_1_2 = keras.layers.Conv2D(filters=320, kernel_size=[3,3], strides=2, padding="same", activation="relu")
        self.bn4_1_2 = keras.layers.BatchNormalization()

        self.conv4_2_1 = keras.layers.Conv2D(filters=192, kernel_size=[1,1], strides=1, padding="same", activation="relu")
        self.bn4_2_1 = keras.layers.BatchNormalization()
        self.conv4_2_2 = keras.layers.Conv2D(filters=192, kernel_size=[1,7], strides=1, padding="same", activation="relu")
        self.bn4_2_2 = keras.layers.BatchNormalization()
        self.conv4_2_3 = keras.layers.Conv2D(filters=192, kernel_size=[7,1], strides=1, padding="same", activation="relu")
        self.bn4_2_3 = keras.layers.BatchNormalization()
        self.conv4_2_4 = keras.layers.Conv2D(filters=192, kernel_size=[3,3], strides=2, padding="same", activation="relu")
        self.bn4_2_4 = keras.layers.BatchNormalization()

        self.maxpool4_3 = keras.layers.MaxPool2D(pool_size=[3,3], strides=2, padding="same")


        self.stage1 = keras.Sequential([
            # [None 32 32 3] -> [None 32 32 32]
            keras.layers.Conv2D(filters=32, kernel_size=[3,3], strides=1, padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            # [None 32 32 32] -> [None 32 32 32]
            keras.layers.Conv2D(filters=32, kernel_size=[3,3], strides=1, padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            # [None 32 32 32] -> [None 32 32 64]
            keras.layers.Conv2D(filters=64, kernel_size=[3,3], strides=1, padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            # [None 32 32 64] -> [None 16 16 64]
            keras.layers.MaxPool2D(pool_size=[3,3], strides=2, padding="same"),
            # [None 16 16 64] -> [None 16 16 80] -> [None 16 16 192] -> [None 8 8 192]
            keras.layers.Conv2D(filters=80, kernel_size=[3,3], strides=1, padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Conv2D(filters=192, kernel_size=[3,3], strides=1, padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.MaxPool2D(pool_size=[3,3], strides=2, padding="same"),
        ])

        self.stage2 = keras.Sequential([
            # [None 8 8 192] -> [None 8 8 256] -> [None 8 8 288] -> [None 8 8 288]
            InceptionV2([64, 48, 64, 64, 96, 32]),
            InceptionV2([64, 48, 64, 64, 96, 64]),
            InceptionV2([64, 48, 64, 64, 96, 64]),
        ])

        self.stage3 = keras.Sequential([
            # [None 4 4 768] -> [None 4 4 768]
            InceptionV3_1([192, 128]),
            InceptionV3_1([192, 160]),
            InceptionV3_1([192, 160]),
            InceptionV3_1([192, 192]),
        ])

        self.stage4 = keras.Sequential([
            # [None 2 2 1280] -> [None 2 2 2048]
            InceptionV3_2(),
            InceptionV3_2(),
        ])

        self.stage5 = keras.Sequential([
            # [None 2 2 2048] -> [None 1 1 2048] -> [None 1 1 1000]
            keras.layers.MaxPool2D(pool_size=[2,2], strides=2, padding="same"),
            keras.layers.Dropout(0.3),
            keras.layers.Conv2D(filters=1000, kernel_size=[1,1], strides=1, padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
        ])

        self.loss_func = keras.losses.CategoricalCrossentropy(from_logits=True)
        self.opt = keras.optimizers.Adam(0.003)

    def call(self, inputs, training=None, mask=None):
        # stage1
        out = self.stage1(inputs)
        # stage2
        out = self.stage2(out)
        # stage3
        # [None 8 8 288] -> [None 4 4 768]
        out3_1 = self.bn3_1(self.conv3_1(out))
        out3_2 = self.bn3_2_3(self.conv3_2_3(self.bn3_2_2(self.conv3_2_2(self.bn3_2_1(self.conv3_2_1(out))))))
        out3_3 = self.maxpool3_1(out)
        out = tf.concat((out3_1, out3_2, out3_3), axis=3)
        out = self.stage3(out)
        # stage4
        # [None 4 4 768] -> [None 2 2 1280]
        out4_1 = self.bn4_1_2(self.conv4_1_2(self.bn4_1_1(self.conv4_1_1(out))))
        out4_2 = self.bn4_2_4(self.conv4_2_4(self.bn4_2_3(self.conv4_2_3(self.bn4_2_2(self.conv4_2_2(self.bn4_2_1(self.conv4_2_1(out))))))))
        out4_3 = self.maxpool4_3(out)
        out = tf.concat((out4_1, out4_2, out4_3), axis=3)
        out = self.stage4(out)
        # stage5
        out = self.stage5(out)
        # dense
        out = keras.layers.Flatten()(out)
        out = keras.layers.Dropout(0.3)(out)
        out = keras.layers.Dense(10)(out)
        return out

def train():
    set_soft_gpu(True)
    dbtrain, dbtest = load_cifar10_datasets(128)
    model = GoogLenetV3()
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