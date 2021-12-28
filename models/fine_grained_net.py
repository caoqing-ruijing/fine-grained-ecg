import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

from .networks import build_backbone, build_final_layer


################ Weakly Supervised Bilinear Attention Network ################

class CenterLayer(keras.layers.Layer):
    """ center loss """

    def __init__(self, class_num, part_num, fea_dim, beta_c=0.05, 
                 lambda_c=1.0, task_type="multi_class", loss_reduction=True, **kwargs):
        super().__init__(**kwargs)
        self.class_num = class_num
        self.part_num = part_num
        self.fea_dim = fea_dim
        self.beta_c = beta_c  # the rate for updating center
        self.lambda_c = lambda_c  # the weight for center loss
        self.task_type = task_type
        self.loss_reduction = loss_reduction

    def build(self, input_shape):
        self.centers = self.add_weight(name='centers',
                                       shape=(self.class_num,
                                              self.part_num, self.fea_dim),
                                       initializer='zeros', trainable=True,
                                       aggregation=tf.VariableAggregation.MEAN)
        # super().build(input_shape)

    def call(self, x, training=None):
        x, label = x[0], x[1]
        # assert x.shape == (bs, self.part_num, self.fea_dim)
        if training is None:
            training = K.learning_phase()

        if training:
            if self.task_type=="multi_class":
                label = tf.cast(label, tf.int32)
                # (bs, part_num, fea_dim)
                centers_batch = tf.gather(self.centers, label, axis=0)
                diff = x - centers_batch
            elif self.task_type=="multi_label":
                centers = tf.expand_dims(self.centers, axis=0)
                label = tf.cast(label, tf.float32)
                label = tf.expand_dims(tf.expand_dims(label, -1), -1)
                centers_batch = tf.reduce_mean(centers*label, axis=1)
                # the result seems to be worse when using the below code
                # centers_batch = tf.reduce_sum(centers*label, axis=1)/tf.reduce_sum(label, axis=1)
                diff = x - centers_batch
            else:
                raise NotImplementedError("Unknown TASK_TYPE: %s"%self.task_type)
            if self.loss_reduction:
                center_loss = self.lambda_c * tf.reduce_mean(tf.square(diff))
            else:
                center_loss = self.lambda_c * tf.reduce_mean(tf.square(diff), (1,2))
        else:
            center_loss = -1.

        self.add_loss(center_loss, inputs=True)
        self.add_metric(center_loss, aggregation='mean', name='center_loss')
        self.add_metric(tf.reduce_mean(self.centers),
                        aggregation='mean', name='center_mean')
        self.add_metric(tf.math.reduce_variance(tf.reduce_mean(self.centers, axis=2)),
                        aggregation='mean', name='center_var')

        return x

    def get_config(self):
        config = super().get_config()
        config.update({'part_num': self.part_num,
                       'fea_dim': self.fea_dim,
                       "beta_c": self.beta_c,
                       "class_num": self.class_num})


def norm_fea(x, axis=-1):
    """ signed sqrt and then L2-norm """
    x = K.sign(x) * K.sqrt(K.abs(x) + 1e-9)
    return K.l2_normalize(x, axis=axis)


def ws_ban(input_shape, num_class,
           freeze_backbone=False, weights='imagenet',
           initial_bias=None, final_activation="softmax",
           ws_ban_dict=None, develop_dict=None):
    """ Weakly Supervised Bilinear Attention Network """
    part_num = ws_ban_dict["part_num"]  # 32
    dropout_rate = ws_ban_dict["dropout_rate"]  # 0.2
    beta_c = ws_ban_dict["beta_c"]  # 0.05
    lambda_c = ws_ban_dict["lambda_c"]  # 1.0
    backbone_name = ws_ban_dict["backbone_name"]  # "develop_net"
    extra_layer_name = ws_ban_dict["extra_layer_name"]  # "conv4_block2_out"
    add_center_layer = ws_ban_dict["add_center_layer"] # True
    add_bap_norm = ws_ban_dict["add_bap_norm"]  # True
    add_multi_head_attn = ws_ban_dict["add_multi_head_attn"]  # False
    multi_head_attn_num = ws_ban_dict["multi_head_attn_num"]  # 1
    attn_head_num = ws_ban_dict["attn_head_num"]  # 8
    multi_head_attn_adding = ["multi_head_attn_adding"] # True
    add_norm_dense_after_attn = ws_ban_dict["add_norm_dense_after_attn"] # True
    dropout_rate_after_attn = ws_ban_dict["add_dropout_after_attn"] # -1 means disable
    pool_layer = keras.layers.GlobalAveragePooling2D
    backbone = build_backbone(backbone_name, input_shape,
                              weights=weights, develop_dict=develop_dict)
    if freeze_backbone:
        backbone.trainable = False
    if extra_layer_name is not None:
        target_layer = backbone.get_layer(extra_layer_name)
    else:
        target_layer = backbone
    img = backbone.input
    if final_activation=="softmax":
        label = keras.layers.Input(())
        task_type = "multi_class"
    elif final_activation=="sigmoid":
        label = keras.layers.Input((num_class,), dtype=tf.float32, name="input_label")
        task_type = "multi_label"
    else:
        raise NotImplementedError("Unknow final activation: %s"%final_activation)
    fea_map = target_layer.output
    fea_map_shape = target_layer.output_shape  # (n, H, W, C)
    fea_dim = fea_map_shape[-1]

    attn_maps = keras.layers.Conv2D(part_num, (1, 1), strides=(1, 1), padding='same',
                                    kernel_initializer='glorot_uniform')(fea_map)
    attn_maps = keras.layers.SpatialDropout2D(dropout_rate)(attn_maps)
    attn_maps = tf.split(attn_maps, part_num, axis=-1)  # [a_1, a_2, ... , a_m]

    part_list = []
    for ai in attn_maps:
        part = fea_map * ai  # (n, H, W, C)
        part = pool_layer()(part)  # (n, C)
        part = tf.expand_dims(part, 1)  # (n, 1, C)
        part_list.append(part)
    if len(part_list)>1:
        bap = keras.layers.concatenate(part_list, axis=1)  # (n, m, C)
    else:
        bap = part_list[0]
    if add_center_layer:

        bap = CenterLayer(num_class, part_num, fea_dim,
                          beta_c, lambda_c, task_type)((bap, label))
    if add_bap_norm:
        bap = norm_fea(bap)
    if add_multi_head_attn:
        from .networks import multi_head_attn
        for _ in range(multi_head_attn_num):
            bap_attn, _ = multi_head_attn.MultiHeadAttention(d_model=fea_dim, num_heads=attn_head_num)(bap, k=bap, q=bap)
            if dropout_rate_after_attn!=-1:
                bap_attn = keras.layers.Dropout(dropout_rate_after_attn)(bap_attn)
            if multi_head_attn_adding:
                bap = keras.layers.Add()([bap, bap_attn]) # (n, m, C)
            else:
                bap = bap_attn
            if add_norm_dense_after_attn:
                bap = keras.layers.LayerNormalization(epsilon=1e-6)(bap)
                ffn_output = multi_head_attn.point_wise_feed_forward_network(fea_dim, max(128, fea_dim//4))(bap)  # (batch_size, input_seq_len, d_model)
                if dropout_rate_after_attn!=-1:
                    ffn_output = keras.layers.Dropout(dropout_rate_after_attn)(ffn_output)
                bap = keras.layers.Add()([bap, ffn_output])
                bap = keras.layers.LayerNormalization(epsilon=1e-6)(bap)
    x = keras.layers.Flatten()(bap) # (n, mxC)
    x = build_final_layer(num_class, initial_bias)(x)
    prob = keras.layers.Activation(final_activation)(x)

    model = tf.keras.Model(inputs=[img, label], outputs=prob,
                           name="ws_ban")
    return model


def normalize_attn_maps(attn_maps, norm_method=None, axis=-1):
    if norm_method is None:
        pass
    elif norm_method == "sigmoid":
        attn_maps = tf.keras.activations.sigmoid(attn_maps)
    elif norm_method == "softmax":
        attn_maps = tf.keras.layers.Softmax(axis=axis)(attn_maps)
    else:
        raise NotImplementedError("Unknown 'norm_method': %s" % norm_method)
    return attn_maps


def ws_ban_v2(input_shape, num_class,
              freeze_backbone=False, weights='imagenet',
              initial_bias=None, final_activation="softmax",
              ws_ban_dict=None, develop_dict=None, attr_shape=None):
    """ Spatial Attention and Channel Attention """
    part_num = ws_ban_dict["part_num"]  # 32
    dropout_rate = ws_ban_dict["dropout_rate"]  # 0.2
    beta_c = ws_ban_dict["beta_c"]  # 0.05
    lambda_c = ws_ban_dict["lambda_c"]  # 1.0
    backbone_name = ws_ban_dict["backbone_name"]  # "develop_net"
    extra_layer_name = ws_ban_dict["extra_layer_name"]  # "conv4_block2_out"
    h_norm_method = ws_ban_dict["h_norm_method"]  # "softmax"
    o_norm_method = ws_ban_dict["o_norm_method"]  # "softmax"
    sp_norm_method = ws_ban_dict["sp_norm_method"]  # "sigmoid"
    ch_norm_method = ws_ban_dict["ch_norm_method"]  # "sigmoid"
    add_down_layers = ws_ban_dict["add_down_layers"]  # False
    add_bap_norm = ws_ban_dict["add_bap_norm"]  # True
    add_channel_attn = ws_ban_dict["add_channel_attn"] # False
    add_attr_branch = ws_ban_dict["add_attr_branch"] # False
    gender_index = ws_ban_dict["gender_index"]
    attr_weight = ws_ban_dict["attr_weight"] # the weight of attribute loss
    add_dense_layer = ws_ban_dict["add_dense_layer"] # False
    feamap_as_h = ws_ban_dict["feamap_as_h"] # False
    pool_layer = keras.layers.GlobalAveragePooling2D
    backbone = build_backbone(backbone_name, input_shape,
                              weights=weights, develop_dict=develop_dict)
    if freeze_backbone:
        backbone.trainable = False
    if extra_layer_name is not None:
        target_layer = backbone.get_layer(extra_layer_name)
    else:
        target_layer = backbone
    img = backbone.input
    label = keras.layers.Input(())
    fea_map = target_layer.output  # (bs, H, W, C)
    fea_map_shape = target_layer.output_shape
    fea_dim = fea_map_shape[-1]
    # attention module
    attn_g_ori = keras.layers.Conv2D(part_num, (1, 1), strides=(1, 1), padding='same',
                                     kernel_initializer='glorot_uniform')(fea_map)  # (bs, H, W, m)
    if feamap_as_h:
        attn_h_ori = fea_map
    else:
        attn_h_ori = keras.layers.Conv2D(fea_dim, (1, 1), strides=(1, 1), padding='same',
                                        kernel_initializer='glorot_uniform')(fea_map)  # (bs, H, W, C)
    attn_g, _ = flatten_hw(attn_g_ori)  # (bs, HxW, m)
    attn_h, _ = flatten_hw(attn_h_ori)  # (bs, HxW, C)
    attn_h = normalize_attn_maps(attn_h, h_norm_method)
    attn_g_T = K.permute_dimensions(attn_g, (0, 2, 1))  # (bs, m, HxW)
    attn_h_T = K.permute_dimensions(attn_h, (0, 2, 1))  # (bs, C, HxW)
    fea_map_flatten, _ = flatten_hw(fea_map)  # (bs, HxW, C)
    # spatial attention maps
    attn_o = K.batch_dot(attn_h_T, attn_g)  # (bs, C, m)
    attn_o = normalize_attn_maps(attn_o, o_norm_method, axis=1)
    spatial_attn_maps = K.batch_dot(fea_map_flatten, attn_o)  # (bs, HxW, m)
    shape_tmp = attn_g_ori.shape.as_list()
    shape_tmp[0] = -1
    spatial_attn_maps = K.reshape(
        spatial_attn_maps, shape_tmp)  # (bs, H, W, m)
    spatial_attn_maps = keras.layers.SpatialDropout2D(
        dropout_rate)(spatial_attn_maps)
    spatial_attn_maps = normalize_attn_maps(spatial_attn_maps, sp_norm_method)
    spatial_attn_maps = tf.split(
        spatial_attn_maps, part_num, axis=-1)  # [a_1, a_2, ... , a_m]
    # channel attention vectors
    if add_channel_attn:
        channel_attn_vecs = K.batch_dot(attn_g_T, attn_h)  # (bs, m, C)
        channel_attn_vecs = normalize_attn_maps(channel_attn_vecs, ch_norm_method)
        channel_attn_vecs = tf.expand_dims(
            channel_attn_vecs, axis=1)  # (bs, 1, m, C)
        channel_attn_vecs = tf.split(
            channel_attn_vecs, part_num, axis=2)  # [v_1, v_2, ..., v_m]

    part_list = []
    if add_down_layers:
        shortcut_pool =  keras.layers.MaxPooling2D(1, strides=2)
        down_layer1 = keras.layers.Conv2D(
            max(128, fea_dim//4), (1, 1), strides=1, padding='same')
        bn1 = keras.layers.BatchNormalization(epsilon=1.001e-5)
        relu1 = keras.layers.Activation('relu')
        down_layer2 = keras.layers.Conv2D(
            fea_dim, (3, 3), strides=2, padding='same')
        bn2 = keras.layers.BatchNormalization(epsilon=1.001e-5)
        relu2 = keras.layers.Activation('relu')
    for i in range(part_num):
        part = fea_map * spatial_attn_maps[i] 
        if add_channel_attn:
            part *= channel_attn_vecs[i]  # (n, H, W, C)
        if add_down_layers:
            shortcut = shortcut_pool(part)
            part = down_layer1(part)
            part = relu1(bn1(part))
            part = down_layer2(part)
            part = keras.layers.Add()([shortcut, part])
            part = relu2(bn2(part))
        part_list.append(part)
    # attribute regression or classification
    if add_attr_branch:
        assert attr_shape is not None
        attrs = keras.layers.Input(
            attr_shape, name="input_attrs")  # (bs, attr_num)
        num_attr = attr_shape[0]
        attr_pred_list = []
        emb_dim = max(128, fea_dim//4)
        for i in range(num_attr):
            activation = "sigmoid" if (i == gender_index) else None
            name = "attr_"+str(i)
            _, _, attr_pred = regress_or_classify_head(
                part_list[i], emb_dim, 1, dropout_rate, activation, name)
            attr_pred_list.append(attr_pred)
    for i in range(part_num):
        part = part_list[i]
        part = pool_layer()(part)  # (n, C)
        part = tf.expand_dims(part, 1)  # (n, 1, C)
        part_list[i] = part
    bap = keras.layers.concatenate(part_list, axis=1)  # (n, m, C)
    bap = CenterLayer(num_class, part_num, fea_dim,
                      beta_c, lambda_c)((bap, label))
    if add_bap_norm:
        bap = norm_fea(bap)
    x = keras.layers.Flatten()(bap)
    if add_dense_layer:
        x = keras.layers.Dropout(dropout_rate)(x)
        x = keras.layers.Dense(512)(x)
        x = keras.layers.BatchNormalization(epsilon=1.001e-5)(x)
        x = keras.layers.Activation("relu")(x)
    x = build_final_layer(num_class, initial_bias)(x)
    prob = keras.layers.Activation(final_activation)(x)

    if add_attr_branch:
        model = tf.keras.Model(inputs=[img, attrs, label], outputs=prob,
                            name="ws_ban_v2")
    else:
        model = tf.keras.Model(inputs=[img, label], outputs=prob,
                            name="ws_ban_v2")
    # attribute branch loss
    if add_attr_branch:
        attr_loss = 0
        attr_loss_list = []
        attr_label_list, mask_list = split_attr_label(attrs, num_attr)
        for i in range(num_attr):
            y_true, y_pred, mask = attr_label_list[i], attr_pred_list[i], mask_list[i]
            attr_loss_i = compute_attr_loss(
                y_true, y_pred, mask, i == gender_index)
            attr_loss_i = attr_loss_i * attr_weight / num_attr
            attr_loss += attr_loss_i
            attr_loss_list.append(attr_loss_i)
        model.add_loss(attr_loss)
        for i, attr_loss_i in enumerate(attr_loss_list):
            model.add_metric(attr_loss_i, aggregation='mean',
                            name='attr_%d_loss' % i)
    return model

################ Attribute-Aware Attention Model ################


def regress_or_classify_head(share_fea_map, dim_channel,
                             num_output=1, dropout=0.5,
                             final_activation=None,
                             name=None):
    # conv
    fea_map = keras.layers.Conv2D(
        dim_channel, (1, 1), padding='same')(share_fea_map)
    fea_map = keras.layers.BatchNormalization(epsilon=1.001e-5)(fea_map)
    fea_map = keras.layers.Activation('relu')(fea_map)
    # pool
    pool = keras.layers.GlobalAveragePooling2D(name=name+'_avg_pool')(fea_map)
    pool = keras.layers.BatchNormalization(epsilon=1.001e-5)(pool)
    pool = keras.layers.Activation('relu')(pool)
    # regression or classification
    pool = keras.layers.Dropout(dropout)(pool)
    pred = keras.layers.Dense(num_output)(pool)
    if final_activation is not None:
        pred = keras.layers.Activation(activation=final_activation,
                                       name=name+'_'+final_activation)(pred)
    # (bs, H, W, dim_channel), (bs, dim_channel), (bs, num_output)
    return fea_map, pool, pred


def flatten_hw(fea_map):
    shape_ori = fea_map.shape  # K.shape(fea_map)
    HxW = shape_ori[1]*shape_ori[2]
    target_shape = (-1, HxW, shape_ori[3])
    target_fea_map = K.reshape(fea_map, shape=target_shape)
    return target_fea_map, HxW.value


def split_attr_label(attrs, num_attr, nan_value=-90):
    attr_label_list = tf.split(attrs, num_attr, axis=-1)  # [attr1, attr2, ...]
    mask_list = tf.split(attrs > nan_value, num_attr, axis=-1)
    return attr_label_list, mask_list


def compute_attr_loss(y_true, y_pred, mask, is_gender=False):
    if is_gender:
        attr_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    else:
        attr_loss = tf.keras.losses.MSE(y_true, y_pred)
    mask_ = tf.cast(tf.squeeze(mask, -1), tf.float32)
    attr_loss = tf.reduce_sum(attr_loss*mask_)
    attr_loss /= tf.reduce_sum(mask_)+K.epsilon()
    return attr_loss


def aaam_net(img_shape, fea_shape, num_class,
             freeze_backbone=False, weights='imagenet',
             initial_bias=None, final_activation="sigmoid",
             aaam_dict=None, develop_dict=None):
    """ Attribute-Aware Attention Model 
    refer to https://github.com/iamhankai/attribute-aware-attention
    # TODO: avoid compute loss for testing mode
    """
    emb_dim = aaam_dict["emb_dim"]  # 512
    dropout = aaam_dict["dropout"]  # 0.5
    # 0  # the index of gender attribute, -1 means nonexistent
    gender_index = aaam_dict["gender_index"]
    fuse_region_method = aaam_dict["fuse_region_method"]  # "max" or "mean"
    # 0.5, 0.5
    cls_weight, attr_weight = aaam_dict["cls_weight"], aaam_dict["attr_weight"]
    backbone_name = aaam_dict["backbone_name"]  # "develop_net"
    global_batch_size = aaam_dict["global_batch_size"]
    final_fea_type = aaam_dict["final_fea_type"] # "attr_region", "attr" or "region"

    # build inputs
    backbone = build_backbone(backbone_name, img_shape, weights, develop_dict)
    share_fea_map = backbone.output
    img = backbone.input
    if final_activation=="softmax":
        label = keras.layers.Input(())
    elif final_activation=="sigmoid":
        label = keras.layers.Input((num_class,), dtype=tf.float32, name="input_label")
    else:
        raise NotImplementedError("Unknow final activation: %s"%final_activation)
    attrs = keras.layers.Input(
        fea_shape, name="input_attrs")  # (bs, attr_num)
    num_attr = fea_shape[0]
    # attr_label_list = tf.split(attrs, num_attr, axis=-1)  # [attr1, attr2, ...]
    # classification branch
    cls_fea_map_ori, cls_pool, cls_prob = regress_or_classify_head(
        share_fea_map, emb_dim, num_class, dropout, final_activation, 'classify_branch')
    # attribute regression or classification
    attr_pool_list = []
    attr_pred_list = []
    for i in range(num_attr):
        activation = "sigmoid" if (i == gender_index) else None
        name = "attr_"+str(i)
        _, attr_pool, attr_pred = regress_or_classify_head(
            share_fea_map, emb_dim, 1, dropout, activation, name)
        attr_pool = tf.expand_dims(attr_pool, -1)  # (bs, emb_dim, 1)
        attr_pool_list.append(attr_pool)
        attr_pred_list.append(attr_pred)

    # attention generation
    cls_fea_map, HxW = flatten_hw(cls_fea_map_ori)  # (bs, HxW, emb_dim)
    cls_pool = tf.expand_dims(cls_pool, axis=1)  # (bs, 1, emb_dim)
    region_attn_map_list = []
    attr_score_list = []
    for i in range(num_attr):
        attn_cls = K.batch_dot(cls_fea_map, attr_pool_list[i])  # (bs, HxW, 1)
        attn_attr = K.batch_dot(cls_pool, attr_pool_list[i])  # (bs, 1)
        region_attn_map_list.append(attn_cls)
        attr_score_list.append(attn_attr)

    # regional feature fusion
    region_attn_map = tf.concat(
        region_attn_map_list, axis=-1)  # (bs, HxW, num_attr)
    if fuse_region_method == "max":
        region_attn_map = tf.reduce_max(
            region_attn_map, axis=-1, keepdims=True)  # (bs, HxW, 1)
    elif fuse_region_method == "mean":
        region_attn_map = tf.reduce_mean(
            region_attn_map, axis=-1, keepdims=True)  # (bs, HxW, 1)
    else:
        raise NotImplementedError(
            "%s is not implemented for 'fuse_region_method'" % fuse_region_method)
    # region_attn_map = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(region_attn_map)
    region_attn_map = tf.keras.layers.Activation(
        'sigmoid', name='region_attention')(region_attn_map)
    region_attn_map /= tf.cast(HxW, tf.float32)
    cls_fea_map_ = K.permute_dimensions(
        cls_fea_map, (0, 2, 1))  # (bs, emb_dim, HxW)
    region_fea = K.batch_dot(cls_fea_map_, region_attn_map)  # (bs, emb_dim, 1)
    region_fea = tf.squeeze(region_fea, -1)  # (bs, emb_dim)
    region_fea = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(region_fea)

    # attribute feature fusion
    attr_scores = tf.concat(attr_score_list, axis=-1)  # (bs, num_attr)
    attr_scores = tf.expand_dims(attr_scores, axis=-1)  # (bs, num_attr, 1)
    # attr_scores = tf.keras.layers.BatchNormalization()(attr_scores)
    attr_scores = tf.keras.layers.Activation(
        'sigmoid', name='attr_attention')(attr_scores)
    attr_scores /= float(num_attr)
    attr_fea = tf.concat(attr_pool_list, axis=-1)  # (bs, emb_dim, num_attr)
    attr_fea = K.batch_dot(attr_fea, attr_scores)  # (bs, emb_dim, 1)
    attr_fea = tf.squeeze(attr_fea, -1)  # (bs, emb_dim)
    attr_fea = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(attr_fea)

    # final classification
    if final_fea_type == "attr_region":
        final_fea = tf.concat([attr_fea, region_fea], axis=-1)  # (bs, 2*emb_dim)
    elif final_fea_type == "attr":
        final_fea = attr_fea
    elif final_fea_type == "region":
        final_fea = region_fea
    else:
        raise NotImplementedError("Unknow final_fea_type: %s"%final_fea_type)
    
    final_fea = tf.keras.layers.Activation('relu', name='final_fea')(final_fea)
    final_fea = tf.keras.layers.Dropout(dropout)(final_fea)
    logits = build_final_layer(
        num_class, initial_bias, "final_logits")(final_fea)
    final_prob = tf.keras.layers.Activation(final_activation)(logits)

    model = tf.keras.Model(inputs=[img, attrs, label], outputs=final_prob,
                           name="aaam_net")

    # classification branch loss
    if final_activation=="softmax":
        label_ = tf.squeeze(label)  # for distributed training
        cls_loss = tf.keras.losses.sparse_categorical_crossentropy(
            label_, cls_prob)
        cls_loss = tf.nn.compute_average_loss(
            cls_loss, global_batch_size=global_batch_size)
    elif final_activation=="sigmoid":
        cls_loss = tf.keras.losses.BinaryCrossentropy()(label, cls_prob)
    else:
        raise NotImplementedError("Unknow final activation: %s"%final_activation)
    cls_loss *= cls_weight
    # attribute branch loss
    attr_loss = 0
    attr_loss_list = []
    attr_label_list, mask_list = split_attr_label(attrs, num_attr)
    for i in range(num_attr):
        y_true, y_pred, mask = attr_label_list[i], attr_pred_list[i], mask_list[i]
        attr_loss_i = compute_attr_loss(
            y_true, y_pred, mask, i == gender_index)
        attr_loss_i = attr_loss_i / num_attr * attr_weight
        attr_loss += attr_loss_i
        attr_loss_list.append(attr_loss_i)
    cls_attr_loss = cls_loss+attr_loss
    model.add_loss(cls_attr_loss)
    model.add_metric(cls_loss, aggregation='mean', name='cls_loss')
    for i, attr_loss_i in enumerate(attr_loss_list):
        model.add_metric(attr_loss_i, aggregation='mean',
                         name='attr_%d_loss' % i)
    return model


def region_attn_net(img_shape, num_class,
                    freeze_backbone=False, weights='imagenet',
                    initial_bias=None, final_activation="softmax", 
                    aaam_dict=None, ws_ban_dict=None):
    """ inspired by AAAM """
    emb_dim = aaam_dict["emb_dim"]  # 512
    dropout = aaam_dict["dropout"]  # 0.5
    fuse_region_method = aaam_dict["fuse_region_method"]  # "max" or "mean" or "center"
    backbone_name = aaam_dict["backbone_name"]  # "develop_net"
    num_attr = aaam_dict["num_attr"]
    final_fea_type = aaam_dict["final_fea_type"] # "cls" or "share"
    extra_layer_name = aaam_dict["extra_layer_name"] # None

    # build inputs
    backbone = build_backbone(backbone_name, img_shape, weights)
    if extra_layer_name is not None:
        target_layer = backbone.get_layer(extra_layer_name)
    else:
        target_layer = backbone
    share_fea_map = target_layer.output
    img = backbone.input

    cls_fea_map_ori = generate_fea_map(share_fea_map, emb_dim)
    cls_fea_map, HxW = flatten_hw(cls_fea_map_ori)

    attr_pool_list = []
    for i in range(num_attr):
        attr_fea_map_i = generate_fea_map(share_fea_map, emb_dim)
        attr_pool_i = generate_pool(attr_fea_map_i, dropout)

        attr_pool_i = tf.expand_dims(attr_pool_i, -1)  # (bs, emb_dim, 1)
        attr_pool_list.append(attr_pool_i)

    if final_fea_type == "share":
        share_fea_map_, HxW = flatten_hw(share_fea_map)
        fea_map_ = K.permute_dimensions(share_fea_map_, (0, 2, 1))  # (n, fea_dim, HxW)
    elif final_fea_type == "cls":
        fea_map_ = K.permute_dimensions(
            cls_fea_map, (0, 2, 1))  # (bs, emb_dim, HxW)
    else: 
        raise NotImplementedError(
            "%s is not implemented for 'final_fea_type'" % final_fea_type)

    if fuse_region_method == "center":
        add_center_layer = ws_ban_dict["add_center_layer"]
        if add_center_layer:
            fea_dim = ws_ban_dict["fea_dim"]
            beta_c, lambda_c = ws_ban_dict["beta_c"], ws_ban_dict["lambda_c"]
            if final_activation=="softmax":
                label = keras.layers.Input(())
                task_type = "multi_class"
            elif final_activation=="sigmoid":
                label = keras.layers.Input((num_class,), dtype=tf.float32, name="input_label")
                task_type = "multi_label"
            else:
                raise NotImplementedError("Unknow final activation: %s"%final_activation)
            center_layer = CenterLayer(num_class, num_attr, fea_dim,
                                            beta_c, lambda_c, task_type)
        else:
            center_layer = None

        add_multi_head_attn = ws_ban_dict["add_multi_head_attn"]
        if add_multi_head_attn:
            assert add_center_layer == True
            from .networks import multi_head_attn
            attn_head_num = ws_ban_dict["attn_head_num"]
            multi_head_attn_layer = multi_head_attn.MultiHeadAttention(
                d_model=fea_dim, num_heads=attn_head_num)
            draa = ws_ban_dict["dropout_rate_after_attn"]
            if draa != -1:
                multi_head_dropout = keras.layers.Dropout(draa)
            else:
                multi_head_dropout = None
            add_norm_dense_after_attn = ws_ban_dict["add_norm_dense_after_attn"]
            if add_norm_dense_after_attn:
                multi_head_norm1 = keras.layers.LayerNormalization(epsilon=1e-6)
                multi_head_ffn = multi_head_attn.point_wise_feed_forward_network(
                    fea_dim, max(128, fea_dim//4))
            multi_head_add = keras.layers.Add()
            multi_head_norm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        else:
            multi_head_attn_layer = None
        # region_attn_map_list = []
        part_list = []
        for i in range(num_attr):
            attn_cls = K.batch_dot(cls_fea_map, attr_pool_list[i])  # (n, HxW, 1)
            region_attn_map = tf.keras.layers.Activation('sigmoid')(attn_cls)
            region_attn_map /= tf.cast(HxW, tf.float32)
            region_fea = K.batch_dot(fea_map_, region_attn_map)  # (n, hidden_dim, 1)
            region_fea = tf.squeeze(region_fea, -1)  # (n, hidden_dim)
            region_fea = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(region_fea)
            region_fea = tf.expand_dims(region_fea, 1) # (n, 1, hidden_dim)
            part_list.append(region_fea)
            # region_attn_map_list.append(region_attn_map)

        if len(part_list) > 1:
            # attn_maps = keras.layers.concatenate(region_attn_map_list, axis=2) # (n, HxW, m)
            bap = keras.layers.concatenate(part_list, axis=1)  # (n, m, hidden_dim)
        else:
            # attn_maps = region_attn_map_list[0]
            bap = part_list[0]

        if center_layer is not None:
            bap = center_layer((bap, label))

        if multi_head_attn_layer is not None:
            bap_attn, _ = multi_head_attn_layer(bap, k=bap, q=bap)
            if multi_head_dropout is not None:
                bap_attn = multi_head_dropout(
                    bap_attn)
            bap = multi_head_add([bap, bap_attn])  # (n, m, hidden_dim)
            if add_norm_dense_after_attn:
                bap = multi_head_norm1(bap)
                # (batch_size, input_seq_len, d_model)
                ffn_output = multi_head_ffn(bap)
                if multi_head_dropout is not None:
                    ffn_output = multi_head_dropout(ffn_output)
                bap = multi_head_add([bap, ffn_output])
                bap = multi_head_norm2(bap)
        final_fea = keras.layers.Flatten()(bap)  # (n, mxhidden_dim)
    else:
        region_attn_map_list = []
        for i in range(num_attr):
            attn_cls = K.batch_dot(cls_fea_map, attr_pool_list[i])  # (bs, HxW, 1)
            region_attn_map_list.append(attn_cls)

        # regional feature fusion
        if num_attr>1:
            region_attn_map = tf.concat(
                region_attn_map_list, axis=-1)  # (bs, HxW, num_attr)
            if fuse_region_method == "max":
                region_attn_map = tf.reduce_max(
                    region_attn_map, axis=-1, keepdims=True)  # (bs, HxW, 1)
            elif fuse_region_method == "mean":
                region_attn_map = tf.reduce_mean(
                    region_attn_map, axis=-1, keepdims=True)  # (bs, HxW, 1)
            else:
                raise NotImplementedError(
                    "%s is not implemented for 'fuse_region_method'" % fuse_region_method)
        else:
            region_attn_map = region_attn_map_list[0]
        # region_attn_map = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(region_attn_map)
        region_attn_map = tf.keras.layers.Activation(
            'sigmoid', name='region_attention')(region_attn_map)
        region_attn_map /= tf.cast(HxW, tf.float32)

        region_fea = K.batch_dot(fea_map_, region_attn_map)  # (bs, emb_dim, 1)
        region_fea = tf.squeeze(region_fea, -1)  # (bs, emb_dim)
        region_fea = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(region_fea)

        final_fea = tf.keras.layers.Activation('relu', name='final_fea')(region_fea)
        final_fea = tf.keras.layers.Dropout(dropout)(final_fea)
        
    logits = build_final_layer(
        num_class, initial_bias, "final_logits")(final_fea)
    final_prob = tf.keras.layers.Activation(final_activation)(logits)

    if fuse_region_method=="center" and ws_ban_dict["add_center_layer"]:
        model = tf.keras.Model(inputs=[img, label], outputs=final_prob,
                            name="region_attn_net")
    else:
        model = tf.keras.Model(inputs=img, outputs=final_prob,
                            name="region_attn_net")

    return model


def generate_fea_map(fea_map, dim_channel):
    # conv
    fea_map = keras.layers.Conv2D(
        dim_channel, (1, 1), padding='same')(fea_map)
    fea_map = keras.layers.BatchNormalization(epsilon=1.001e-5)(fea_map)
    fea_map = keras.layers.Activation('relu')(fea_map)
    return fea_map


def generate_pool(fea_map, dropout):
    # pool
    pool = keras.layers.GlobalAveragePooling2D()(fea_map)
    pool = keras.layers.BatchNormalization(epsilon=1.001e-5)(pool)
    pool = keras.layers.Activation('relu')(pool)
    pool = keras.layers.Dropout(dropout)(pool)
    return pool
