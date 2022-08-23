import tensorflow as tf
from tensorflow import keras
from utils import set_soft_gpu, load_cifar10_datasets

class InceptionV2(keras.layers.Layer):

    def __init__(self, filters):
        super(InceptionV2, self).__init__()
        # 第一条路  [1 1]conv
        self.path1 = keras.Sequential([
            keras.layers.Conv2D(filters=filters[0], kernel_size=[1, 1], strides=1, padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
        ])
        # 第二条路  [1 1]conv [3 3]conv
        self.path2 = keras.Sequential([
            keras.layers.Conv2D(filters=filters[1], kernel_size=[1, 1], strides=1, padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Conv2D(filters=filters[2], kernel_size=[3, 3], strides=1, padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
        ])
        # 第三条路  [1 1]conv [3 3]conv [3 3]conv
        self.path3 = keras.Sequential([
            keras.layers.Conv2D(filters=filters[3], kernel_size=[1, 1], strides=1, padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Conv2D(filters=filters[4], kernel_size=[3, 3], strides=1, padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Conv2D(filters=filters[4], kernel_size=[3, 3], strides=1, padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
        ])
        # 第四条路  [3 3]pool [1 1]conv
        self.path4 = keras.Sequential([
            keras.layers.MaxPool2D(pool_size=[3, 3], strides=1, padding="same"),
            keras.layers.Conv2D(filters=filters[5], kernel_size=[1, 1], strides=1, padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
        ])

    def call(self, inputs, **kwargs):
        # out1
        out_1 = self.path1(inputs)
        # out2
        out_2 = self.path2(inputs)
        # out3
        out_3 = self.path3(inputs)
        # out4
        out_4 = self.path4(inputs)
        # concat
        out = tf.concat((out_1,out_2,out_3,out_4), axis=3)
        return out

class InceptionV2_pass_through(keras.layers.Layer):

    def __init__(self, filters):
        super(InceptionV2_pass_through, self).__init__()
        # 第二条路  [1 1]conv [3 3]conv
        self.path2 = keras.Sequential([
            keras.layers.Conv2D(filters=filters[0], kernel_size=[1, 1], strides=1, padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Conv2D(filters=filters[1], kernel_size=[3, 3], strides=1, padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
        ])
        # 第三条路  [1 1]conv [3 3]conv [3 3]conv
        self.path3 = keras.Sequential([
            keras.layers.Conv2D(filters=filters[2], kernel_size=[1, 1], strides=1, padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Conv2D(filters=filters[3], kernel_size=[3, 3], strides=1, padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Conv2D(filters=filters[3], kernel_size=[3, 3], strides=1, padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
        ])
        # 第四条路  [3 3]pool
        self.p4_M = keras.layers.MaxPool2D(pool_size=[3,3], strides=1, padding="same")

    def call(self, inputs, **kwargs):
        # out2
        out_2 = self.path2(inputs)
        # out3
        out_3 = self.path3(inputs)
        # out4
        out_4 = self.p4_M(inputs)
        out = tf.concat((out_2,out_3,out_4), axis=3)
        return out

class GoogLenetV2(keras.Model):

    def __init__(self):
        super(GoogLenetV2, self).__init__()

        self.stage1 = keras.Sequential([
            # [None 32 32 3] -> [None 32 32 64] -> [None 16 16 64]
            keras.layers.Conv2D(filters=64, kernel_size=[7,7], strides=1, padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.MaxPool2D(pool_size=[3,3], strides=2, padding="same"),
        ])

        self.stage2 = keras.Sequential([
            # [None 16 16 64] -> [None 16 16 192] -> [None 16 16 192] -> [None 8 8 192]
            keras.layers.Conv2D(filters=192, kernel_size=[1,1], strides=1, padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Conv2D(filters=192, kernel_size=[3,3], strides=1, padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.MaxPool2D(pool_size=[3,3], strides=2, padding="same"),
        ])

        self.stage3 = keras.Sequential([
            # [None 8 8 192] -> [None 8 8 256] -> [None 8 8 320]
            InceptionV2([64, 64, 64, 64, 96, 32]),
            InceptionV2([64, 64, 96, 64, 96, 64]),
        ])
        self.stage3_pass_through = keras.Sequential([
            # [None 8 8 320] -> [None 8 8 256]
            InceptionV2_pass_through([128, 160, 64, 96])
        ])

        self.stage4 = keras.Sequential([
            # [None 4 4 576] -> [None 4 4 576] -> [None 4 4 576] -> [None 4 4 576] -> [None 4 4 576]
            InceptionV2([224, 64, 96, 96, 128, 128]),
            InceptionV2([192, 96, 128, 96, 128, 128]),
            InceptionV2([160, 128, 160, 128, 160, 96]),
            InceptionV2([96, 128, 192, 160, 192, 96]),
        ])
        self.stage4_pass_through = keras.Sequential([
            # [None 4 4 576] -> [None 4 4 448]
            InceptionV2_pass_through([128, 192, 192, 256])
        ])

        self.stage5 = keras.Sequential([
            # [None 2 2 1024] -> [None 2 2 1024] -> [None 2 2 1024] -> [None 1 1 1024] -> [None 1024]
            InceptionV2([352, 192, 320, 160, 224, 128]),
            InceptionV2([352, 192, 320, 192, 224, 128]),
            keras.layers.GlobalAvgPool2D(),
            keras.layers.Flatten(),
        ])

        self.pool = keras.layers.MaxPool2D(pool_size=[3,3], strides=2, padding="same")
        self.dropout = keras.layers.Dropout(0.4)
        self.dense1 = keras.layers.Dense(10)

        self.loss_func = keras.losses.CategoricalCrossentropy(from_logits=True)
        self.opt = keras.optimizers.Adam(0.0002)

    def call(self, inputs, training=None, mask=None):
        # [None 32 32 3] -> [None 16 16 64]
        out = self.stage1(inputs)
        # [None 16 16 64] -> [None 8 8 192]
        out = self.stage2(out)
        # out31 [None 8 8 192] -> [None 8 8 320]  out32 [None 8 8 320] -> [None 8 8 256]   out concat((out31, out32))
        # pool [None 8 8 576] -> [None 4 4 576]
        out31 = self.stage3(out)
        out32 = self.stage3_pass_through(out31)
        out = tf.concat((out31, out32), axis=3)
        out = self.pool(out)
        # out41 [None 4 4 576] -> [None 4 4 576]  out42 [None 4 4 576] -> [None 4 4 448]   out concat((out41, out42))
        # pool [None 4 4 1024] -> [None 2 2 1024]
        out41 = self.stage4(out)
        out42 = self.stage4_pass_through(out41)
        out = tf.concat((out41, out42), axis=3)
        out = self.pool(out)
        # [None 2 2 1024] -> [None 1024]
        out = self.stage5(out)
        out = self.dropout(out)
        out = self.dense1(out)
        return out

def train():
    set_soft_gpu(True)
    dbtrain, dbtest = load_cifar10_datasets(128)
    model = GoogLenetV2()
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
