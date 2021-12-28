import os
import numpy as np
from functools import partial
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, Dense, Dropout, Flatten
from tensorflow.keras import applications as kapp
from tensorflow.python.keras.applications import keras_modules_injection

from .image_transform import preprocess_symbolic_input
import sys
# sys.path.append("../")
from public_code import keras_applications, SelfAttentionLayer, multi_head_attn


@keras_modules_injection
def EfficientNetB0(*args, **kwargs):
    return keras_applications.efficientnet.EfficientNetB0(*args, **kwargs)


@keras_modules_injection
def EfficientNetB1(*args, **kwargs):
    return keras_applications.efficientnet.EfficientNetB1(*args, **kwargs)


@keras_modules_injection
def EfficientNetB2(*args, **kwargs):
    return keras_applications.efficientnet.EfficientNetB2(*args, **kwargs)


@keras_modules_injection
def ResNet34V2(*args, **kwargs):
    return keras_applications.resnet_common.ResNet34V2(*args, **kwargs)


def extract_dict_from_config(cfg, prefix):
    target_dict = {}
    for key, item in cfg._asdict().items():
        if key.startswith(prefix):
            key = key.replace(prefix, '').lower()
            target_dict[key] = item
    if len(target_dict) == 0:
        target_dict = None
    return target_dict


def build_model(cfg=None, initial_bias=None):
    img_shape = (cfg.IMG_HEIGHT, cfg.IMG_WIDTH, cfg.IMG_CHANNELS)
    fea_shape = (len(cfg.FEA_NAMES),)
    nc = len(cfg.LABEL_NAMES)
    fb = cfg.FREEZE_BACKBONE
    fjs = cfg.FEATURE_JOINT_STRATEGY
    ws = None if cfg.FROM_SCRATCH else cfg.PRETRAINED_WEIGHT  # 'imagenet'
    sf = cfg.SQUEEZE_FEA_FOR_BILINEAR
    cm = cfg.CONCAT_MAXPOOL
    develop_dict = extract_dict_from_config(cfg, "DEV_")
    # final activation
    if cfg.LOGITS_FOR_LOSS:
        fa = None
    elif cfg.TASK_TYPE == "multi_label":
        # for multi-label binary classification
        fa = "sigmoid"
    else:
        fa = "softmax"

    if cfg.NETWORK == "cnn_rnn_with_attn":
        if cfg.CRA_RNN_TYPE=="conv_lstm":
            from .cnn_convlstm import build_cnn_rnn_with_attn
        else:
            from .cnn_rnn import build_cnn_rnn_with_attn
        cra_dict = extract_dict_from_config(cfg, "CRA_")
        center_attn_dict = extract_dict_from_config(cfg, "CRA_CA_")
        multi_head_attn_dict = extract_dict_from_config(cfg, "CRA_MHA_")
        model = build_cnn_rnn_with_attn(img_shape, ws, nc, cra_dict, (center_attn_dict, multi_head_attn_dict))
    else:
        print('{} not support'.format(cfg.NETWORK))
        return None
    return model


def build_img_preprocess(cfg):
    """ different pretrained model has different preprocess method for images
    Args:
        net_name: the name the pretrained model
        imagenet_norm: whether using ImageNet normalization for "torch" or "caffe" mode
    Return:
        the preprocess function

    refer https://github.com/keras-team/keras-applications/tree/master/keras_applications
    """
    net_name, imagenet_norm = cfg.NETWORK, cfg.IMAGENET_NORM
    norm_dict = {
        "torch": ["dense_net", "efficient_net"],
        "tf": ["mobile_net", "res_net", "xception", "inception_v3"],  # v2
        # skip preprocess
        "skip": ["simple_cnn", "inaturalist", "develop_net",
                 "ws_ban", "ws_ban_v2", "aaam_net", "res_net34", 
                 "vgg", "cnn_rnn_with_attn"]
    }
    # convert key->item to item->key
    norm_dict_tmp = {}
    for key, item in norm_dict.items():
        for i in item:
            norm_dict_tmp[i] = key

    pre_fn = None
    if cfg.FROM_SCRATCH:
        pre_fn = False
    else:
        for key, item in norm_dict_tmp.items():
            if key in net_name:
                if item == "skip":
                    pre_fn = False
                else:
                    pre_fn = partial(preprocess_symbolic_input,
                                     mode=item, imagenet_norm=imagenet_norm)
    if pre_fn is None:
        raise NotImplementedError(
            "%s is not supported for net_name" % net_name)
    return pre_fn


def load_model_weight(model, cfg, weight_path=None):
    if weight_path is None:
        weight_path = os.path.join(cfg.CKPT_DIR, "best_model.h5")
    try:
        model.load_weights(weight_path)
        print("SimpleCNN or only finetune last layer")
    except ValueError:
        model.trainable = True
        model.load_weights(weight_path)
        print("finetune all layers")
    return model


def compute_initial_bias(df_train, cfg):
    initial_bias = cfg.FINAL_LAYER_INITIAL_BIAS
    if initial_bias:
        assert cfg.TASK_TYPE == "multi_class"
        if isinstance(initial_bias, bool):
            count_dict = df_train[cfg.TASK_NAME].value_counts()
            total = float(len(df_train))
            initial_bias = np.log(
                [count_dict[i]/total for i in range(len(cfg.LABEL_NAMES))])
    else:
        initial_bias = None
    return initial_bias


def build_final_layer(num_class, output_bias=None, name=None):
    """ Classifier
    Args:
        num_class: the number of the classes
        output_bias: a list of  initial bias, default is zero initialization
    """
    if output_bias is not None:
        output_bias = keras.initializers.Constant(output_bias)
    else:
        output_bias = "zeros"
    name = "logits" if name is None else name
    final_layer = Dense(num_class, activation=None,
                        bias_initializer=output_bias, name=name)
    return final_layer


def build_backbone(net_name, input_shape, weights='imagenet', develop_dict=None):
    backbone_dict = {
        "dense_net121": kapp.DenseNet121,
        "mobile_net": kapp.MobileNetV2,
        "vgg16": kapp.VGG16,
        "efficient_netb0": EfficientNetB0,
        "efficient_netb1": EfficientNetB1,
        "efficient_netb2": EfficientNetB2,
        "res_net34": ResNet34V2,
        "res_net50": kapp.ResNet50V2,
        "xception": kapp.Xception,
        "inception_v3": kapp.InceptionV3,
        "develop_net": None
    }
    # load local pretrained weights
    weight_path = False
    if weights not in ["imagenet", None]:
        weight_path = weights
        weights = None
    if net_name in backbone_dict.keys():
        if net_name == "develop_net":
            backbone = develop_net(
                input_shape, include_top=False, develop_dict=develop_dict)
        else:
            backbone = backbone_dict[net_name](
                input_shape=input_shape,
                include_top=False,
                weights=weights)
    else:
        raise NotImplementedError(
            "%s is not supported for net_name" % net_name)
    if weight_path:
        backbone.load_weights(weight_path)
        print("load pretrained weight from %s"%weight_path)
    return backbone


def combine_imgfea_numfea(imgfea, numfea, joint_strategy):
    if joint_strategy == "concat_early":
        x = keras.layers.concatenate([imgfea, numfea])
        x = Dense(512)(x)
        x = keras.layers.ELU()(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
    elif joint_strategy == "concat_later":
        x = Dense(128)(numfea)
        x = keras.layers.ELU()(x)
        x = BatchNormalization()(x)
        # x = Dropout(0.4)(x)
        x = keras.layers.concatenate([imgfea, x])
        x = Dense(512)(x)
        x = keras.layers.ELU()(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
    elif joint_strategy == "concat_only":
        x = keras.layers.concatenate([imgfea, numfea])
    else:
        raise NotImplementedError(
            "%s is not supproted for joint_strategy" % joint_strategy)
    return x


def pretrained_net(net_name, input_shape, num_class,
                   freeze_backbone=False, weights='imagenet',
                   initial_bias=None, concat_maxpool=False,
                   final_activation="softmax", develop_dict=None):
    backbone = build_backbone(net_name, input_shape,
                              weights=weights, develop_dict=develop_dict)
    if freeze_backbone:
        backbone.trainable = False
    img = backbone.input
    x = backbone.output
    x1 = keras.layers.GlobalAveragePooling2D()(x)
    if concat_maxpool:
        x2 = keras.layers.GlobalMaxPooling2D()(x)
        x1 = keras.layers.concatenate([x1, x2])
    x = build_final_layer(num_class, initial_bias)(x1)
    prob = keras.layers.Activation(final_activation)(x)
    model = keras.Model(inputs=img, outputs=prob,
                        name=net_name)

    return model


def pretrained_net_with_fea(net_name, img_shape,
                            fea_shape, num_class,
                            freeze_backbone=False,
                            joint_strategy="concat_later",
                            weights='imagenet',
                            initial_bias=None,
                            concat_maxpool=False,
                            final_activation="softmax",
                            develop_dict=None):
    backbone = build_backbone(net_name, img_shape, weights, develop_dict)
    img = backbone.input
    fea = keras.layers.Input(fea_shape, name="input_fea")
    if freeze_backbone:
        backbone.trainable = False

    x = backbone.output
    x1 = keras.layers.GlobalAveragePooling2D()(x)
    if concat_maxpool:
        x2 = keras.layers.GlobalMaxPooling2D()(x)
        x1 = keras.layers.concatenate([x1, x2])
    x = combine_imgfea_numfea(x1, fea, joint_strategy)
    x = build_final_layer(num_class, initial_bias)(x)
    prob = keras.layers.Activation(final_activation)(x)

    model = keras.Model(inputs=[img, fea], outputs=prob,
                        name='%s_with_fea' % net_name)

    return model


def group_conv(x, kernel_size, stride, filters, groups):
    c = filters // groups
    x = keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = keras.layers.DepthwiseConv2D(kernel_size, strides=stride, depth_multiplier=c,
                                     use_bias=False)(x)
    kernel = np.zeros((1, 1, filters * c, filters), dtype=np.float32)
    for i in range(filters):
        start = (i // c) * c * c + i % c
        end = start + c * c
        kernel[:, :, start:end:c, i] = 1.
    x = keras.layers.Conv2D(filters, 1, use_bias=False, trainable=False,
                            kernel_initializer={'class_name': 'Constant',
                                                'config': {'value': kernel}})(x)
    return x


def simple_cnn(input_shape, num_class=None, include_top=True, initial_bias=None,
               depthwise=False, final_activation="softmax"):
    model = keras.models.Sequential(name="simple_cnn")
    # groups = input_shape[-1]
    if depthwise:
        model.add(keras.layers.DepthwiseConv2D((3, 3), strides=(1, 1), depth_multiplier=6,
                                               input_shape=input_shape))
    else:
        model.add(Conv2D(16, (3, 3), strides=(1, 1),
                         input_shape=input_shape, kernel_initializer='glorot_uniform'))
    model.add(keras.layers.ELU())
    model.add(BatchNormalization())

    model.add(Conv2D(16, (3, 3), strides=(1, 1),
                     kernel_initializer='glorot_uniform'))
    model.add(keras.layers.ELU())
    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(16, (3, 3), strides=(1, 1),
                     kernel_initializer='glorot_uniform'))
    model.add(keras.layers.ELU())
    model.add(BatchNormalization())

    model.add(Conv2D(32, (3, 3), strides=(1, 1),
                     kernel_initializer='glorot_uniform'))
    model.add(keras.layers.ELU())
    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(32, (3, 3), strides=(1, 1),
                     kernel_initializer='glorot_uniform'))
    model.add(keras.layers.ELU())
    model.add(BatchNormalization())

    model.add(Conv2D(16, (3, 3), strides=(1, 1),
                     kernel_initializer='glorot_uniform'))
    model.add(keras.layers.ELU())
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    if include_top:
        model.add(Dense(256))
        model.add(keras.layers.ELU())
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        model.add(build_final_layer(num_class, initial_bias))
        model.add(keras.layers.Activation(final_activation))

    return model


def simple_cnn_with_fea(img_shape, fea_shape, num_class,
                        joint_strategy, initial_bias=None,
                        wider=False, depthwise=False, final_activation="softmax"):
    if wider:
        net_name = "simple_wider_cnn_with_fea"
        backbone = simple_wider_cnn(img_shape, include_top=False)
    else:
        net_name = "simple_cnn_with_fea"
        backbone = simple_cnn(
            img_shape, include_top=False, depthwise=depthwise)
    img = backbone.input
    fea = keras.layers.Input(fea_shape, name="input_fea")

    fcn = keras.models.Sequential(name="fully_conneted_net")
    fcn.add(Dense(256))
    fcn.add(keras.layers.ELU())
    fcn.add(BatchNormalization())
    # fcn.add(Dropout(0.4))

    x = backbone.output
    x = fcn(x)
    x = combine_imgfea_numfea(x, fea, joint_strategy)
    x = build_final_layer(num_class, initial_bias)(x)
    prob = keras.layers.Activation(final_activation)(x)

    model = keras.Model(inputs=[img, fea], outputs=prob,
                        name=net_name)
    return model


def simple_wider_cnn(input_shape=None, num_class=None, include_top=True,
                     initial_bias=None, kernel_size=(3, 3), input_tensor=None,
                     concat_maxpool=False, final_activation="softmax"):
    activation = keras.layers.ELU
    if input_tensor is None:
        image = keras.layers.Input(shape=input_shape)
    else:
        image = input_tensor
    x = Conv2D(32, (5, 5), strides=(2, 2), padding='same',
               kernel_initializer='glorot_uniform')(image)
    x = activation()(x)
    x = BatchNormalization(epsilon=1.001e-5)(x)
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = Conv2D(32, kernel_size, strides=(1, 1), padding='same',
               kernel_initializer='glorot_uniform')(x)
    x = activation()(x)
    x = BatchNormalization(epsilon=1.001e-5)(x)

    x = Conv2D(64, kernel_size, strides=(2, 2), padding='same',
               kernel_initializer='glorot_uniform')(x)
    x = activation()(x)
    x = BatchNormalization(epsilon=1.001e-5)(x)

    x = Conv2D(64, kernel_size, strides=(1, 1), padding='same',
               kernel_initializer='glorot_uniform')(x)
    x = activation()(x)
    x = BatchNormalization(epsilon=1.001e-5)(x)

    x = Conv2D(128, kernel_size, strides=(2, 2), padding='same',
               kernel_initializer='glorot_uniform')(x)
    x = activation()(x)
    x = BatchNormalization(epsilon=1.001e-5)(x)

    x = Conv2D(128, kernel_size, strides=(1, 1), padding='same',
               kernel_initializer='glorot_uniform')(x)
    x = activation()(x)
    x = BatchNormalization(epsilon=1.001e-5)(x)

    x = Conv2D(256, kernel_size, strides=(1, 1), padding='same',
               kernel_initializer='glorot_uniform')(x)
    x = activation()(x)
    x = BatchNormalization(epsilon=1.001e-5)(x)
    x1 = keras.layers.GlobalAveragePooling2D()(x)
    if concat_maxpool:
        x2 = keras.layers.GlobalMaxPooling2D()(x)
        x1 = keras.layers.concatenate([x1, x2])
    if include_top:
        x = build_final_layer(num_class, initial_bias)(x1)
        prob = keras.layers.Activation(final_activation)(x)
    else:
        prob = x1

    model = keras.Model(inputs=image, outputs=prob,
                        name="simple_wider_cnn")
    return model


@keras_modules_injection
def stack2(*args, **kwargs):
    return keras_applications.resnet_common.ResStack2(*args, **kwargs)


def develop_net(input_shape=None, num_class=None, include_top=True,
                initial_bias=None, input_tensor=None, concat_maxpool=False,
                final_activation="softmax", develop_dict=None):
    self_attn1 = develop_dict["self_attn1"]  # False
    self_attn2 = develop_dict["self_attn2"]  # False
    activation = develop_dict["activation"]  # 'relu'
    depthwise = develop_dict["depth_wise"]  # False
    attn_type = develop_dict["self_attn_type"]  # self_attn_with_leads"
    block_dims = develop_dict["block_dims"]  # [32, 64, 64, 128, 128]
    block_i = 0

    if attn_type == "self_attn_origin":
        attn_layer = SelfAttentionLayer.SelfAttention
    elif attn_type == "self_attn_with_leads":
        attn_layer = SelfAttentionLayer.SelfAttentionWithLeads
    elif attn_type == "double_attn_with_leads":
        attn_layer = SelfAttentionLayer.DoubleAttentionWithLeads
    elif attn_type == "self_double_attn_with_leads":
        attn_layer = SelfAttentionLayer.SelfDoubleAttentionWithLeads
    else:
        raise NotImplementedError("Unknow 'attn_type': %s" % attn_layer)

    bn_axis = 3 if keras.backend.image_data_format() == 'channels_last' else 1
    if input_tensor is None:
        image = keras.layers.Input(shape=input_shape)
    else:
        image = input_tensor
    if depthwise:
        split_num = 1  # disable split
        x = keras.layers.DepthwiseConv2D((5, 5), strides=(2, 2), depth_multiplier=block_dims[block_i],
                                         padding='same', kernel_initializer='glorot_uniform')(image)  # [b, h, w, 32*3]
    else:
        split_num = 3
        x = Conv2D(block_dims[block_i], (5, 5), strides=(2, 2), padding='same',
                   kernel_initializer='glorot_uniform')(image)
    x = keras.layers.ZeroPadding2D(
        padding=((1, 1), (1, 1)), name='pool1_pad')(x)
    x = keras.layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)
    if self_attn1:
        nc = block_dims[block_i]*3 if depthwise else block_dims[block_i]
        scale = 3 if depthwise else 1
        x = attn_layer(nc, scale, split_num=split_num)(x)
    block_i += 1
    x = stack2(x, block_dims[block_i], 3, activation=activation, name='conv2')
    if self_attn2:
        x = attn_layer(block_dims[block_i]*4, split_num=split_num)(x)
    block_i += 1
    x = stack2(x, block_dims[block_i], 3, activation=activation, name='conv3')
    block_i += 1
    x = stack2(x, block_dims[block_i], 3, activation=activation, name='conv4')
    block_i += 1
    x = stack2(x, block_dims[block_i], 2, stride1=1,
               activation=activation, name='conv5')
    x = keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                        name='post_bn')(x)
    x = keras.layers.Activation(activation, name='post_relu')(x)
    if include_top:
        x1 = keras.layers.GlobalAveragePooling2D()(x)
        if concat_maxpool:
            x2 = keras.layers.GlobalMaxPooling2D()(x)
            x1 = keras.layers.concatenate([x1, x2])
        x = build_final_layer(num_class, initial_bias)(x1)
        prob = keras.layers.Activation(final_activation)(x)
    else:
        prob = x

    model = keras.Model(inputs=image, outputs=prob,
                        name="develop_net")
    return model

# TODO:  pass each row of ECG iteratively for develop net
