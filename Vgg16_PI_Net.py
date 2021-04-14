from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.applications.vgg16 import VGG16
import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

dataset = "cherry"
model_type = "VGG16"
img_size = (224, 224, 3)
class_num = 88


def channel_attention_block_new(x1):
    mlp1= layers.Dense(2048,
                      activation='relu',
                      kernel_initializer='he_normal',
                      use_bias=True,
                      bias_initializer='zeros')
    mlp2 = layers.Dense(8012,
                      activation='sigmoid',
                      kernel_initializer='he_normal',
                      use_bias=True,
                      bias_initializer='zeros')
    att_x = mlp1(x1)
    att_x = mlp2(att_x)
    x = layers.multiply([x1, att_x])
    return x


def PINet_CIFAR10(model_input):
    ## model
    initial_conv_width = 3
    initial_stride = 1
    initial_pool_width = 3
    initial_pool_stride = 2
    use_global_pooling = True

    x = layers.Conv2D(
        128,
        initial_conv_width,
        strides=initial_stride,
        padding="same")(model_input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(
        pool_size=initial_pool_width,
        strides=initial_pool_stride,
        padding="same")(x)

    x = layers.Conv2D(
        256,
        initial_conv_width,
        strides=initial_stride,
        padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(
        pool_size=initial_pool_width,
        strides=initial_pool_stride,
        padding="same")(x)

    x = layers.Conv2D(
        512,
        initial_conv_width,
        strides=initial_stride,
        padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(
        pool_size=initial_pool_width,
        strides=initial_pool_stride,
        padding="same")(x)

    x = layers.Conv2D(
        1024,
        initial_conv_width,
        strides=initial_stride,
        padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    if use_global_pooling:
        x = layers.GlobalAveragePooling2D()(x)

    x_logits1 = layers.Dense(2500, activation="relu")(x)

    x_logits1_reshape = layers.Reshape((1, 50, 50))(x_logits1)

    x_logits1_reshape = layers.Permute((2, 3, 1))(x_logits1_reshape)

    x_logits2 = layers.Conv2DTranspose(3, 50, strides=initial_stride, padding="same")(x_logits1_reshape)
    x_logits2 = layers.BatchNormalization()(x_logits2)
    x_logits2 = layers.Activation("relu")(x_logits2)

    model_output = layers.Flatten()(x_logits2)
    model = models.Model(model_input, model_output)
    return model


def get_vgg16():
    input_tensor = layers.Input(shape=img_size)
    base_model = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor, input_shape=img_size, pooling='max')
    x = base_model.output
    x = layers.Dense(512, activation='relu', name='fc1')(x)
    y = layers.Dense(class_num, activation='softmax', name='prediction')(x)
    model = models.Model(inputs=input_tensor, outputs=y)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model


def get_model_with_attention():
    input_tensor = layers.Input(shape=img_size)
    base_model = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor, input_shape=img_size, pooling='max')
    x = base_model.output
    x = layers.BatchNormalization()(x)

    pi_net = PINet_CIFAR10(input_tensor)
    pi_net.load_weights("./PI-Net_CIFAR10.h5")
    pi_net.trainable = False
    x1 = pi_net.output
    x1 = layers.BatchNormalization()(x1)
    x = layers.Concatenate()([x, x1])
    x = channel_attention_block_new(x)
    x = layers.Dense(8012, activation='relu', name='fc1')(x)
    y = layers.Dense(class_num, activation='softmax', name='prediction')(x)

    model = models.Model(inputs=input_tensor, outputs=y)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model


def get_model():
    input_tensor = layers.Input(shape=img_size)
    base_model = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor, input_shape=img_size, pooling='max')
    x = base_model.output
    x = layers.BatchNormalization()(x)

    pi_net = PINet_CIFAR10(input_tensor)
    pi_net.load_weights("./PI-Net_CIFAR10.h5")
    pi_net.trainable = False
    x1 = pi_net.output
    x1 = layers.BatchNormalization()(x1)

    x = layers.Concatenate()([x, x1])
    x = layers.Dense(8012, activation='relu', name='fc1')(x)
    y = layers.Dense(class_num, activation='softmax', name='prediction')(x)

    model = models.Model(inputs=input_tensor, outputs=y)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model


from torch_model import get_dataset_file_list_from_csv
from skimage import io as skio
from skimage.transform import resize
import numpy as np
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from tqdm import tqdm
from sklearn.metrics import classification_report

cherry_img_path = r"/home/zyp/MFCIS/dataset/cherry/cherry_jpg256_cultivar100"
folio_img_path = r"/home/zyp/project/mobile_tp_net/Folio_256"

ids = np.loadtxt("./{}_id.txt".format(dataset))
id_map = dict()
for d in ids:
    id_map[int(d[0])] = int(d[1])


def load_img(file_list, y_list):
    img_x = []
    y = []
    for index, f in enumerate(tqdm(file_list)):
        if f.endswith("a0.png") or f.endswith("a1.png") or f.endswith("a2.png"):
            continue
        if dataset == "cherry":
            img = skio.imread(os.path.join(cherry_img_path, f))
        elif dataset == "folio":
            img = skio.imread(os.path.join(folio_img_path, f))
        else:
            img = skio.imread(f)
        img = resize(img, img_size)
        img_x.append(img)
        class_idx = id_map[int(y_list[index])]
        y.append(class_idx)
    img_x = np.asarray(img_x)
    y = np.asarray(y)
    return img_x, y


if __name__ == "__main__":
    if model_type == "VGG16":
        model = get_vgg16()
    elif model_type == "VGG16_PI_NET":
        model = get_model()
    elif model_type == "VGG16_PI_NET_WITH_ATTENTION":
        model = get_model_with_attention()
    train_x_list, train_y_list = get_dataset_file_list_from_csv("./{}_train_file_0.csv".format(dataset))
    val_x_list, val_y_list = get_dataset_file_list_from_csv("./{}_val_file_0.csv".format(dataset))
    test_x_list, test_y_list = get_dataset_file_list_from_csv("./{}_test_file_0.csv".format(dataset))
    train_img_x, train_y = load_img(train_x_list, train_y_list)
    val_img_x, val_y = load_img(val_x_list, val_y_list)
    test_img_x, test_y = load_img(test_x_list, test_y_list)
    lr_adjust = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.5,
                                  patience=5,
                                  min_lr=1e-7)

    save_best_weight = ModelCheckpoint('{}_weight_{}.hdf5'.format(model_type, dataset),
                                       monitor='val_loss',
                                       verbose=1,
                                       save_best_only=True,
                                       mode='auto',
                                       save_weights_only=True)
    train_y_one_hot = to_categorical(train_y)
    val_y_one_hot = to_categorical(val_y)
    test_y_one_hot = to_categorical(test_y)
    model.fit(train_img_x, train_y_one_hot, batch_size=32, epochs=70, validation_split=0.1,
              callbacks=[lr_adjust, save_best_weight])
    K.clear_session()

    if model_type == "VGG16":
        model2 = get_vgg16()
    elif model_type == "VGG16_PI_NET":
        model2 = get_model()
    elif model_type == "VGG16_PI_NET_WITH_ATTENTION":
        model2 = get_model_with_attention()
    model2.load_weights('{}_weight_{}.hdf5'.format(model_type, dataset))
    score = model2.evaluate(test_img_x, test_y_one_hot)
    pre_final = model2.predict(test_img_x)
    y_test_label = np.array([np.argmax(d) for d in test_y_one_hot])
    y_pre_label = np.array([np.argmax(d) for d in pre_final])
    report = classification_report(y_test_label, y_pre_label)
    print(report)
    print(score)
