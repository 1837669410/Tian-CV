import tensorflow as tf
from tensorflow import keras
from utils import set_soft_gpu, load_cifar10_datasets

class Basic_block(keras.layers.Layer):

    def __init__(self, n_conv, filters, kernel_size, model_label=None):
        super(Basic_block, self,).__init__()
        self.model = self.build_block(n_conv, filters, kernel_size, model_label)

    def call(self, inputs, **kwargs):
        return self.model(inputs)

    def build_block(self, n_conv, filters, kernel_size, model_label):
        self.vgg_block = keras.Sequential([])
        for i in range(n_conv):
            self.vgg_block.add(keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=1, padding="same"))
        if model_label == "vgg16-1":
            self.vgg_block.add(keras.layers.Conv2D(filters=filters, kernel_size=[1,1], strides=1, padding="same"))
        self.vgg_block.add(keras.layers.MaxPool2D(pool_size=[2,2], strides=2, padding="same"))
        self.vgg_block.add(keras.layers.ReLU())
        return self.vgg_block

class VGG(keras.Model):
    '''
    五种VGG模型的配置信息，第一个参数是当前卷积层的卷积个数，第二个参数是filter，第三个参数是卷积核
    vgg11: one_layer: [1, 64, [3,3]] two_layer: [1, 128, [3,3]] three_layer: [2, 256, [3,3]] four_layer: [2, 512, [3,3]] five_layer: [2, 512, [3,3]]
    vgg13: one_layer: [2, 64, [3,3]] two_layer: [2, 128, [3,3]] three_layer: [2, 256, [3,3]] four_layer: [2, 512, [3,3]] five_layer: [2, 512, [3,3]]
    vgg16-1: one_layer: [2, 64, [3,3]] two_layer: [2, 128, [3,3]] three_layer: [2, 256, [3,3], "vgg16"] four_layer: [2, 512, [3,3], "vgg16"] five_layer: [2, 512, [3,3], "vgg16"]
    vgg16-3: one_layer: [2, 64, [3,3]] two_layer: [2, 128, [3,3]] three_layer: [3, 256, [3,3]] four_layer: [3, 512, [3,3]] five_layer: [3, 512, [3,3]]
    vgg16-3: one_layer: [2, 64, [3,3]] two_layer: [2, 128, [3,3]] three_layer: [4, 256, [3,3]] four_layer: [4, 512, [3,3]] five_layer: [4, 512, [3,3]]
    '''

    def __init__(self, model_label="vgg11"):
        super(VGG, self).__init__()
        self.model_label = model_label
        self.model = self.build_model()

        self.loss_func = keras.losses.CategoricalCrossentropy(from_logits=True)
        self.opt = keras.optimizers.Adam(0.0001)

    def call(self, inputs, training=None, mask=None):
        return self.model.call(inputs)

    def build_model(self):
        if self.model_label == "vgg11":
            model = keras.Sequential([
                # [None 32 32 3] -> [None 32 32 64] -> [None 16 16 64]
                Basic_block(1, 64, [3,3]),
                # [None 16 16 64] -> [None 16 16 128] -> [None 8 8 128]
                Basic_block(1, 128, [3,3]),
                # [None 8 8 128] -> [None 8 8 256] -> [None 4 4 256]
                Basic_block(2, 256, [3,3]),
                # [None 4 4 256] -> [None 4 4 512] -> [None 2 2 512]
                Basic_block(2, 512, [3,3]),
                # [None 2 2 512] -> [None 2 2 512] -> [None 1 1 512]
                Basic_block(2, 512, [3,3]),
                # Flatten
                keras.layers.Flatten(),
                # [None 512] -> [None 256] -> [None 128] -> [None 10]
                keras.layers.Dense(256),
                keras.layers.ReLU(),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(128),
                keras.layers.ReLU(),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(10)
            ], name=self.model_label)
            model.build(input_shape=[None, 32, 32, 3])
            model.summary()
            return model

        elif self.model_label == "vgg13":
            model = keras.Sequential([
                # [None 32 32 3] -> [None 32 32 64] -> [None 16 16 64]
                Basic_block(2, 64, [3,3]),
                # [None 16 16 64] -> [None 16 16 128] -> [None 8 8 128]
                Basic_block(2, 128, [3,3]),
                # [None 8 8 128] -> [None 8 8 256] -> [None 4 4 256]
                Basic_block(2, 256, [3,3]),
                # [None 4 4 256] -> [None 4 4 512] -> [None 2 2 512]
                Basic_block(2, 512, [3,3]),
                # [None 2 2 512] -> [None 2 2 512] -> [None 1 1 512]
                Basic_block(2, 512, [3,3]),
                # Flatten
                keras.layers.Flatten(),
                # [None 512] -> [None 256] -> [None 128] -> [None 10]
                keras.layers.Dense(256),
                keras.layers.ReLU(),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(128),
                keras.layers.ReLU(),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(10)
            ], name=self.model_label)
            model.build(input_shape=[None, 32, 32, 3])
            model.summary()
            return model

        elif self.model_label == "vgg16-1":
            model = keras.Sequential([
                # [None 32 32 3] -> [None 32 32 64] -> [None 16 16 64]
                Basic_block(2, 64, [3,3]),
                # [None 16 16 64] -> [None 16 16 128] -> [None 8 8 128]
                Basic_block(2, 128, [3,3]),
                # [None 8 8 128] -> [None 8 8 256] -> [None 4 4 256]
                Basic_block(2, 256, [3,3], model_label=self.model_label),
                # [None 4 4 256] -> [None 4 4 512] -> [None 2 2 512]
                Basic_block(2, 512, [3,3], model_label=self.model_label),
                # [None 2 2 512] -> [None 2 2 512] -> [None 1 1 512]
                Basic_block(2, 512, [3,3], model_label=self.model_label),
                # Flatten
                keras.layers.Flatten(),
                # [None 512] -> [None 256] -> [None 128] -> [None 10]
                keras.layers.Dense(256),
                keras.layers.ReLU(),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(128),
                keras.layers.ReLU(),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(10)
            ], name=self.model_label)
            model.build(input_shape=[None, 32, 32, 3])
            model.summary()
            return model

        elif self.model_label == "vgg16-3":
            model = keras.Sequential([
                # [None 32 32 3] -> [None 32 32 64] -> [None 16 16 64]
                Basic_block(2, 64, [3,3]),
                # [None 16 16 64] -> [None 16 16 128] -> [None 8 8 128]
                Basic_block(2, 128, [3,3]),
                # [None 8 8 128] -> [None 8 8 256] -> [None 4 4 256]
                Basic_block(3, 256, [3,3]),
                # [None 4 4 256] -> [None 4 4 512] -> [None 2 2 512]
                Basic_block(3, 512, [3,3]),
                # [None 2 2 512] -> [None 2 2 512] -> [None 1 1 512]
                Basic_block(3, 512, [3,3]),
                # Flatten
                keras.layers.Flatten(),
                # [None 512] -> [None 256] -> [None 128] -> [None 10]
                keras.layers.Dense(256),
                keras.layers.ReLU(),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(128),
                keras.layers.ReLU(),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(10)
            ], name=self.model_label)
            model.build(input_shape=[None, 32, 32, 3])
            model.summary()
            return model

        elif self.model_label == "vgg19":
            model = keras.Sequential([
                # [None 32 32 3] -> [None 32 32 64] -> [None 16 16 64]
                Basic_block(2, 64, [3,3]),
                # [None 16 16 64] -> [None 16 16 128] -> [None 8 8 128]
                Basic_block(2, 128, [3,3]),
                # [None 8 8 128] -> [None 8 8 256] -> [None 4 4 256]
                Basic_block(4, 256, [3,3]),
                # [None 4 4 256] -> [None 4 4 512] -> [None 2 2 512]
                Basic_block(4, 512, [3,3]),
                # [None 2 2 512] -> [None 2 2 512] -> [None 1 1 512]
                Basic_block(4, 512, [3,3]),
                # Flatten
                keras.layers.Flatten(),
                # [None 512] -> [None 256] -> [None 128] -> [None 10]
                keras.layers.Dense(256),
                keras.layers.ReLU(),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(128),
                keras.layers.ReLU(),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(10)
            ], name=self.model_label)
            model.build(input_shape=[None, 32, 32, 3])
            model.summary()
            return model

def train():
    set_soft_gpu(True)
    dbtrain, dbtest = load_cifar10_datasets(128)
    model = VGG(model_label="vgg19")
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


