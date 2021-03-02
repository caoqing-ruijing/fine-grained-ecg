import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.python.keras.applications import keras_modules_injection

from my_utils import networks
import sys
sys.path.append("../")
from public_code import keras_applications, SelfAttentionLayer, multi_head_attn


@keras_modules_injection
def ResNet34V2OneDOriginal(*args, **kwargs):
    return keras_applications.resnet_common_1d.ResNet34V2OneDOriginal(*args, **kwargs)


@keras_modules_injection
def ResNet34V2OneD(*args, **kwargs):
    return keras_applications.resnet_common_1d.ResNet34V2OneD(*args, **kwargs)


@keras_modules_injection
def ResNet34V2OneDBigStride(*args, **kwargs):
    return keras_applications.resnet_common_1d.ResNet34V2OneDBigStride(*args, **kwargs)


def build_backbone(backbone_name, input_shape, net_dict):
    fks = net_dict["first_kernel_size"]
    if backbone_name == "resnet34v2_1d_origianl":
        backbone = ResNet34V2OneDOriginal(input_shape=input_shape, include_top=False)
    elif backbone_name == "resnet34v2_1d":
        backbone = ResNet34V2OneD(input_shape=input_shape, include_top=False, first_kernel_size=fks)
    elif backbone_name == "resnet34v2_1d_big_stride":
        backbone = ResNet34V2OneDBigStride(input_shape=input_shape, include_top=False, first_kernel_size=fks)
    else:
        raise NotImplementedError("%s is not supproted" % backbone_name)
    return backbone


def well_know_net(net_name, input_shape, num_class,
                   freeze_backbone=False, initial_bias=None, 
                   concat_maxpool=False, final_activation="softmax", net_dict=None):
    backbone = build_backbone(net_name, input_shape, net_dict)
    if freeze_backbone:
        backbone.trainable = False
    img = backbone.input
    x = backbone.output
    x1 = keras.layers.GlobalAveragePooling1D()(x)
    if concat_maxpool:
        x2 = keras.layers.GlobalMaxPooling1D()(x)
        x1 = keras.layers.concatenate([x1, x2])
    x = networks.build_final_layer(num_class, initial_bias)(x1)
    prob = keras.layers.Activation(final_activation)(x)
    model = keras.Model(inputs=img, outputs=prob,
                        name=net_name)

    return model


def build_model_for_digital(cfg):
        input_shape = (cfg.IMG_WIDTH, cfg.IMG_CHANNELS)
        nc = len(cfg.LABEL_NAMES)
        fb = cfg.FREEZE_BACKBONE
        if cfg.LOGITS_FOR_LOSS:
            fa = None
        elif cfg.TASK_TYPE == "multi_label":
            # for multi-label binary classification
            fa = "sigmoid"
        else:
            fa = "softmax"
        if cfg.NETWORK == "ecgNet":
            sys.path.append("../ecg_multi-label-classification_tf/")
            from utils.tools import load_cfg
            from models.ecgNet import ecgNet

            defaut_cfg_path = '../ecg_multi-label-classification_tf/configs/default_cfg.json'
            args_config = './configs/digital_v1/RuiJing_refine82_tfdata_json_signalLarge_4x10_sigmoid_cosLR.json'
            model_cfg = load_cfg(defaut_cfg_path,args_config)
            model_cfg = model_cfg._replace(n_classes=len(cfg.LABEL_NAMES))
            model = ecgNet(model_cfg,verbose=False).build()
        elif cfg.NETWORK == "region_attn_net_1d":
            aaam_dict = networks.extract_dict_from_config(cfg, "AAAM_")
            backbone_dict = networks.extract_dict_from_config(cfg, "DEV_")
            model = region_attn_net_1d(input_shape, nc, fb, backbone_dict, None, fa, aaam_dict)
        else:
            cm = cfg.CONCAT_MAXPOOL
            net_dict = networks.extract_dict_from_config(cfg, "DEV_")
            model = well_know_net(cfg.NETWORK, input_shape, nc, fb, None, cm, fa, net_dict)

        return model


def region_attn_net_1d(input_shape, num_class,
                    freeze_backbone=False, backbone_dict=None,
                    initial_bias=None, final_activation="softmax", 
                    aaam_dict=None):
    """ inspired by AAAM """
    emb_dim = aaam_dict["emb_dim"]  # 512
    dropout = aaam_dict["dropout"]  # 0.5
    fuse_region_method = aaam_dict["fuse_region_method"]  # "max" or "mean"
    backbone_name = aaam_dict["backbone_name"]  # "develop_net"
    num_attr = aaam_dict["num_attr"]
    final_fea_type = aaam_dict["final_fea_type"] # "cls" or "share"
    extra_layer_name = aaam_dict["extra_layer_name"] # None

    # build inputs
    # backbone = build_backbone(backbone_name, input_shape, weights)
    backbone = build_backbone(backbone_name, input_shape, backbone_dict)
    if extra_layer_name is not None:
        target_layer = backbone.get_layer(extra_layer_name)
    else:
        target_layer = backbone
    share_fea_map = target_layer.output # (bs, length, fea_dim)
    img = backbone.input # (bs, 5000, 4)

    cls_fea_map = generate_fea_map(share_fea_map, emb_dim) # (bs, length, emb_dim)
    # cls_fea_map, HxW = flatten_hw(cls_fea_map_ori)

    attr_pool_list = []
    for i in range(num_attr):
        attr_fea_map_i = generate_fea_map(share_fea_map, emb_dim) # (bs, length, emb_dim)
        attr_pool_i = generate_pool(attr_fea_map_i, dropout) # (bs, emb_dim)

        attr_pool_i = tf.expand_dims(attr_pool_i, -1)  # (bs, emb_dim, 1)
        attr_pool_list.append(attr_pool_i)

    if final_fea_type == "share":
        # share_fea_map_, HxW = flatten_hw(share_fea_map)
        fea_map_ = K.permute_dimensions(share_fea_map, (0, 2, 1))  # (n, fea_dim, length)
    elif final_fea_type == "cls":
        fea_map_ = K.permute_dimensions(cls_fea_map, (0, 2, 1))  # (bs, emb_dim, length)
    else: 
        raise NotImplementedError(
            "%s is not implemented for 'final_fea_type'" % final_fea_type)

    region_attn_map_list = []
    for i in range(num_attr):
        attn_cls = K.batch_dot(cls_fea_map, attr_pool_list[i])  # (bs, length, 1)
        region_attn_map_list.append(attn_cls)

    # regional feature fusion
    if num_attr>1:
        region_attn_map = tf.concat(
            region_attn_map_list, axis=-1)  # (bs, length, num_attr)
        if fuse_region_method == "max":
            region_attn_map = tf.reduce_max(
                region_attn_map, axis=-1, keepdims=True)  # (bs, length, 1)
        elif fuse_region_method == "mean":
            region_attn_map = tf.reduce_mean(
                region_attn_map, axis=-1, keepdims=True)  # (bs, length, 1)
        else:
            raise NotImplementedError(
                "%s is not implemented for 'fuse_region_method'" % fuse_region_method)
    else:
        region_attn_map = region_attn_map_list[0]
    region_attn_map = tf.keras.layers.Activation(
        'sigmoid', name='region_attention')(region_attn_map)
    region_attn_map /= tf.cast(region_attn_map.shape[1].value, tf.float32) # (bs, length, 1)

    region_fea = K.batch_dot(fea_map_, region_attn_map)  # (bs, emb_dim, 1)
    region_fea = tf.squeeze(region_fea, -1)  # (bs, emb_dim)
    region_fea = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(region_fea)

    final_fea = tf.keras.layers.Activation('relu', name='final_fea')(region_fea)
    final_fea = tf.keras.layers.Dropout(dropout)(final_fea)
        
    logits = networks.build_final_layer(
        num_class, initial_bias, "final_logits")(final_fea)
    final_prob = tf.keras.layers.Activation(final_activation)(logits)

    model = tf.keras.Model(inputs=img, outputs=final_prob,
                        name="region_attn_net_1d")

    return model


def generate_fea_map(fea_map, dim_channel):
    # conv
    fea_map = keras.layers.Conv1D(
        dim_channel, 1, padding='same')(fea_map)
    fea_map = keras.layers.BatchNormalization(epsilon=1.001e-5)(fea_map)
    fea_map = keras.layers.Activation('relu')(fea_map)
    return fea_map


def generate_pool(fea_map, dropout):
    # pool
    pool = keras.layers.GlobalAveragePooling1D()(fea_map)
    pool = keras.layers.BatchNormalization(epsilon=1.001e-5)(pool)
    pool = keras.layers.Activation('relu')(pool)
    pool = keras.layers.Dropout(dropout)(pool)
    return pool