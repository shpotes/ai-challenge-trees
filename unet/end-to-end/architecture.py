import math
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D, Dropout, UpSampling2D, Input
from tensorflow.keras.layers import Cropping2D, concatenate
from tensorflow.keras.applications.vgg16 import VGG16

def crop_shape(down, up):
    ch = int(down[1] - up[1])
    cw = int(down[2] - up[2])
    ch1, ch2 = ch // 2, int(math.ceil(ch / 2))
    cw1, cw2 = cw // 2, int(math.ceil(cw / 2))

    return (ch1, ch2), (cw1, cw2)


def get_shape(x):
    return tuple(x.get_shape().as_list())


def conv2d_block(inputs, filters=16):
    c = inputs
    for _ in range(2):
        c = Conv2D(filters, (3,3), activation='relu', padding='valid') (c)
    return c


def vanilla_unet(in_shape):
    input = Input((in_shape, in_shape, 3))
    x = input

    # Downsampling path
    down_layers = []
    filters = 64
    for _ in range(4):
        x = conv2d_block(x, filters)
        down_layers.append(x)
        x = MaxPooling2D((2, 2), strides=2) (x)
        filters *= 2 # Number of filters doubled with each layer

    x = conv2d_block(x, filters)

    for conv in reversed(down_layers):
        filters //= 2
        x = Conv2DTranspose(filters, (2, 2), strides=(2, 2),
                            padding='same') (x)


        ch, cw = crop_shape(get_shape(conv), get_shape(x))
        conv = Cropping2D((ch, cw)) (conv)

        x = concatenate([x, conv])
        x = conv2d_block(x, filters)

    output = Conv2D(1, (1, 1), activation='sigmoid') (x)
    return Model(input, output)


def vgg_unet(in_shape, FREEZING_DEPTH=5, DROPPING_DEPTH=7):
    input_tensor = Input(shape=(in_shape, in_shape, 3))
    feature_extractor = VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)

    # Remove last layers
    for _ in range(DROPPING_DEPTH):
        feature_extractor._layers.pop() 

    # freeze model
    for i in feature_extractor._layers[:FREEZING_DEPTH]:
        i.trainable = False

    LAST_POOL = list(reversed(list(map(lambda x: 'pool' in x.name, feature_extractor._layers)))).index(True)
    x = feature_extractor.layers[-1].output

    for i, conv in reversed(list(enumerate(feature_extractor._layers[:-LAST_POOL]))):
        if 'conv' in conv.name:
            x = Conv2D(
                filters=conv.filters,
                kernel_size=conv.kernel_size,
                strides=conv.strides,
                padding=conv.padding,
                activation='relu') (x)
        if 'pool' in conv.name:
            x = UpSampling2D(conv.pool_size) (x)
            x = concatenate([x, feature_extractor._layers[i-1].output])

    output = Conv2D(1, (1, 1), activation='sigmoid') (x)
    return Model(input_tensor, output)
