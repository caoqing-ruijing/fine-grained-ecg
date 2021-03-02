import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

from .networks import build_backbone, build_final_layer
from .fine_grained_net import CenterLayer, norm_fea #, flatten_hw
from .cnn_rnn import build_backbone_with_2outs, call_layers



class CenterSpatialAttention(tf.keras.Model):
    def __init__(self, hidden_dim, num_class, center_attn_dict, multi_head_attn_dict):
        super(CenterSpatialAttention, self).__init__()
        self.part_num = center_attn_dict["part_num"]
        self.fea_conv = keras.layers.Conv2D(hidden_dim, (1, 1), strides=(1, 1), padding='same',
                                            kernel_initializer='glorot_uniform')
        self.attn_conv = keras.layers.Conv2D(self.part_num, (1, 1), strides=(1, 1), padding='same',
                                             kernel_initializer='glorot_uniform')
        dropout_rate = center_attn_dict["dropout_rate"]  # 0.2
        self.attn_dropout = keras.layers.SpatialDropout2D(dropout_rate)
        # self.pool_layer = keras.layers.GlobalAveragePooling2D()

        fea_dim = center_attn_dict["fea_dim"]
        # gate_for_bap = center_attn_dict["gate_for_bap"]
        # if gate_for_bap:
        #     self.gate_fc = keras.layers.Dense(fea_dim)
        # else:
        #     self.gate_fc = None
        self.use_spatial_hidden = center_attn_dict["use_spatial_hidden"]
        if self.use_spatial_hidden:
            self.fea_h, self.fea_w = center_attn_dict["fea_size"] # H x W
            self.fuse_hidden_method = center_attn_dict["fuse_hidden_method"] # "add" or "concat"
            self.hidden_spatial_fc = keras.layers.Dense(self.fea_h * self.fea_w)
        add_center_layer = center_attn_dict["add_center_layer"]
        if add_center_layer:
            beta_c, lambda_c = center_attn_dict["beta_c"], center_attn_dict["lambda_c"]
            self.center_layer = CenterLayer(num_class, self.part_num, fea_dim,
                                            beta_c, lambda_c, "multi_class", False)
        else:
            self.center_layer = None
        add_multi_head_attn = multi_head_attn_dict["add_multi_head_attn"]
        if add_multi_head_attn:
            from .networks import multi_head_attn
            attn_head_num = multi_head_attn_dict["attn_head_num"]
            self.multi_head_attn_layer = multi_head_attn.MultiHeadAttention(
                d_model=fea_dim, num_heads=attn_head_num)
            draa = multi_head_attn_dict["dropout_rate_after_attn"]
            if draa != -1:
                self.multi_head_dropout = keras.layers.Dropout(draa)
            else:
                self.multi_head_dropout = None
            self.add_norm_dense_after_attn = multi_head_attn_dict["add_norm_dense_after_attn"]
            # if self.add_norm_dense_after_attn:
            #     self.multi_head_norm1 = keras.layers.LayerNormalization(epsilon=1e-6)
            #     self.multi_head_ffn = multi_head_attn.point_wise_feed_forward_network(
            #         fea_dim, max(128, fea_dim//4))
            self.multi_head_add = keras.layers.Add()
            # self.multi_head_norm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        else:
            self.multi_head_attn_layer = None

    def call(self, fea_map, hidden, label=None, training=False):
        # (b, H, W, C), where C = fea_dim+hidden_dim
        fea_map = self.fea_conv(fea_map)
        fea_map_hidden = tf.concat([fea_map, hidden], axis=-1)
        # fea_map_hidden = self.fea_conv(fea_map) + hidden_
        attn_maps = self.attn_conv(fea_map_hidden)
        attn_maps = self.attn_dropout(attn_maps, training=training)
        attn_maps = tf.split(attn_maps, self.part_num,
                             axis=-1)  # [a_1, a_2, ... , a_m]
        # if self.gate_fc is not None:
        #     gate = tf.math.sigmoid(self.gate_fc(hidden))
        # else:
        #     gate = None

        part_list = []
        for ai in attn_maps:
            part = fea_map * ai  # (n, H, W, C)
            # part = self.pool_layer(part)  # (n, C)
            # if gate is not None:
            #     part *= gate
            part = tf.expand_dims(part, 1)  # (n, 1, H, W, C)
            part_list.append(part)
        if len(part_list) > 1:
            bap = keras.layers.concatenate(part_list, axis=1)  # (n, m, H, W, C)
        else:
            bap = part_list[0]
        if self.center_layer is not None:
            bap_mean = tf.reduce_mean(bap, axis=(2,3)) # (n, m, C)
            _ = self.center_layer((bap_mean, label), training=training)
        bap = norm_fea(bap) # (n, m, H, W, C)
        if self.multi_head_attn_layer is not None:
            bap_new = tf.transpose(bap, perm=[0,2,3,1,4]) # (n, H, W, m, C)
            bap_shape =  tf.shape(bap_new)
            bap_new =  tf.reshape(bap_new, (-1, self.part_num, bap_shape[-1])) # (nHW, m, C)
            bap_attn, _ = self.multi_head_attn_layer(bap_new, k=bap_new, q=bap_new)
            if self.multi_head_dropout is not None:
                bap_attn = self.multi_head_dropout(
                    bap_attn, training=training)
            bap_attn = tf.reshape(bap_attn, bap_shape) # (n, H, W, m, C)
            bap_attn = tf.transpose(bap_attn, perm=[0,3,1,2,4]) # (n, m, H, W, C)
            bap = self.multi_head_add([bap, bap_attn])  # (n, m, H, W, C)
            # if self.add_norm_dense_after_attn:
            #     bap = self.multi_head_norm1(bap, training=training)
            #     # (batch_size, input_seq_len, d_model)
            #     ffn_output = self.multi_head_ffn(bap)
            #     if self.multi_head_dropout is not None:
            #         ffn_output = self.multi_head_dropout(
            #             ffn_output, training=training)
            #     bap = self.multi_head_add([bap, ffn_output])
            #     bap = self.multi_head_norm2(bap, training=training)
        # bap = keras.layers.Flatten()(bap)  # (n, mxC)
        bap = tf.squeeze(tf.concat(tf.split(bap, self.part_num, 1), axis=-1), 1) # (n, H, W, mxC)
        return bap, attn_maps


def flatten_hw_v2(fea_map):
    shape_ori = fea_map.shape  # K.shape(fea_map)
    H, W = shape_ori[1], shape_ori[2]
    HxW = H*W
    target_shape = (-1, HxW, shape_ori[3])
    target_fea_map = K.reshape(fea_map, shape=target_shape)
    return target_fea_map, (H.value, W.value)


class CNN_Encoder(tf.keras.Model):
        # Since you have already extracted the features and dumped it using pickle
        # This encoder passes those features through a Fully connected layer
    def __init__(self, input_shape, backbone_name, hidden_dim,
                 extra_layer_name=None, weights="imagenet", 
                 add_linear=False, region_attn=False):
        super(CNN_Encoder, self).__init__()
        self.backbone = build_backbone_with_2outs(backbone_name, input_shape,
                                                  extra_layer_name, weights)
        self.add_linear = add_linear
        # self.pool = tf.keras.layers.GlobalAveragePooling2D()
        if self.add_linear:
            self.fc = tf.keras.layers.Dense(hidden_dim)

        self.region_attn = region_attn
        if self.region_attn:
            self.conv_bn_relu_list = []
            for _ in range(2):
                tmp = [keras.layers.Conv2D(hidden_dim, (1, 1), padding='same'),
                    keras.layers.BatchNormalization(epsilon=1.001e-5),
                    keras.layers.Activation('relu')]
                self.conv_bn_relu_list.append(tmp)

            tmp = [keras.layers.GlobalAveragePooling2D(),
                keras.layers.BatchNormalization(epsilon=1.001e-5),
                keras.layers.Activation('relu'),
                keras.layers.Dropout(0.2)]
            self.pool_bn_relu_dropout = tmp
            self.region_bn = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)
            self.sigmoid = tf.keras.layers.Activation('sigmoid')

    def call(self, x, training=False):
        fea_map, fea_vec = self.backbone(x, training=training)
        fea_map = fea_vec # keep same size
        if self.region_attn:
            cls_fea_map_ori = call_layers(self.conv_bn_relu_list[1], fea_vec, training)
            cls_fea_map, (H, W) = flatten_hw_v2(cls_fea_map_ori)

            attr_fea_map_i = call_layers(self.conv_bn_relu_list[0], fea_vec, training)
            attr_pool_i = call_layers(self.pool_bn_relu_dropout, attr_fea_map_i, training)
            attr_pool_i = tf.expand_dims(attr_pool_i, -1)  # (n, hidden_dim, 1)

            # TODO: `fea_map` -> `cls_fea_map`
            # fea_map_ = K.permute_dimensions(cls_fea_map, (0, 2, 1))  # (n, hidden_dim, HxW)
            # fea_map_, HxW = flatten_hw(fea_vec)   # (n, HxW, fea_dim)
            # fea_map_ = K.permute_dimensions(fea_map_, (0, 2, 1))  # (n, fea_dim, HxW)
            attn_cls = K.batch_dot(cls_fea_map, attr_pool_i)  # (n, HxW, 1)
            attn_cls = tf.reshape(attn_cls, (-1, H, W, 1))
            region_attn_map = self.sigmoid(attn_cls)
            # region_attn_map /= tf.cast(HxW, tf.float32)
            # region_fea = K.batch_dot(fea_map_, region_attn_map)  # (n, hidden_dim, 1)
            region_fea = fea_vec * region_attn_map 
            # region_fea = tf.squeeze(region_fea, -1)  # (n, hidden_dim)
            fea_vec = self.region_bn(region_fea, training=training)
        else:
            if self.add_linear:
                fea_vec = self.fc(fea_vec)  # (n, embedding_dim)
                fea_vec = tf.nn.relu(fea_vec)
            # fea_vec = self.pool(fea_vec)
        
        return fea_map, fea_vec


class RNN_Decoder(tf.keras.Model):
    def __init__(self, fea_size, hidden_dim, vocab_size, num_label,
                 attn_type="Bahdanau", rnn_type="gru",
                 center_attn_dicts=None, add_layer_norm=False,
                 kernel_size=3):
        super(RNN_Decoder, self).__init__()
        self.units = hidden_dim
        self.rnn_type = rnn_type
        self.fea_h, self.fea_w = fea_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, self.fea_h*self.fea_w)
        if self.rnn_type == "conv_gru":
            pass
            # TODO: convGRU
            # self.rnn = tf.keras.layers.GRU(
            #     self.units, return_sequences=True, return_state=True,
            #     recurrent_initializer='glorot_uniform'
            # )
        elif self.rnn_type == "conv_lstm":
            self.rnn = tf.keras.layers.ConvLSTM2D(
                self.units, kernel_size, (1,1), "same", return_sequences=True,
                 return_state=True, recurrent_initializer='glorot_uniform'
            )
        # self.fc1 = tf.keras.layers.Dense(self.units)
        if add_layer_norm:
            self.norm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        else:
            self.norm2 = None
        self.fc2 = tf.keras.layers.Dense(num_label)

        self.attn_type = attn_type
        if self.attn_type == "center_spatial_attn":
            self.attention = CenterSpatialAttention(hidden_dim, num_label,
                                             center_attn_dicts[0], 
                                             center_attn_dicts[1]) # TODO: gate=True
        else:
            self.attention = None
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()

    def call(self, x, features, hidden, label=None, training=False):
        # x shape after passing through embedding == (batch_size, 1, embedding_dim), 
        # where embedding_dim = HxW
        x = self.embedding(x)
        x = tf.reshape(x, (-1, 1, self.fea_h, self.fea_w, 1)) # (batch_size, 1,  H, W, 1)

        if self.attention is not None:
            # defining attention as a separate model
            if "center" in self.attn_type:
                context_vec, attn_weights = self.attention(
                    features, hidden[0], label, training)
            else:
                raise NotImplementedError("attention module")
                # context_vec, attn_weights = self.attention(features, hidden[0])

            # x shape after concatenation == (batch_size, 1, H, W, 1 + fea_size)
            x = tf.concat([tf.expand_dims(context_vec, 1), x], axis=-1)
        else:
            attn_weights = None

        # passing the concatenated vector to the GRU
        outputs = self.rnn(x, initial_state=hidden)
        output, state = outputs[0], outputs[1:]

        # shape == (batch_size, max_length, hidden_size)
        # x = self.fc1(output)

        # x shape == (batch_size * max_length, hidden_size)
        # x = tf.reshape(x, (-1, x.shape[2]))
        x = tf.squeeze(output, 1) # (batch_size, H, W, 1 + hidden_size)
        x = self.global_pool(x)
        if self.norm2 is not None:
            x = self.norm2(x, training=training)

        # output shape == (batch_size, vocab)
        x = self.fc2(x)

        return x, state, attn_weights

    def reset_state(self, batch_size=1, initial_state='zeros'):
        if initial_state == 'zeros':
            initial_state = tf.zeros((batch_size, self.fea_h, self.fea_w, self.units))

        if "gru" in self.rnn_type:
            return (initial_state,)
        elif "lstm" in self.rnn_type:
            return (initial_state, initial_state)


def build_cnn_rnn_with_attn(img_shape, ws, nc, cra_dict, center_attn_dicts=None):
    backbone_name = cra_dict["backbone_name"]
    extra_layer_name = cra_dict["extra_layer_name"]  # None
    rnn_type = cra_dict["rnn_type"]  # "gru" or "lstm"
    attn_type = cra_dict["attn_type"]  # "Bahdanau", "center_attn", "center_attn_v2"
    hidden_dim = cra_dict["hidden_dim"]  # 512
    init_state = cra_dict["init_hidden_state"]  # "zero" or "img_fea"
    add_layer_norm = cra_dict["add_layer_norm_after_rnn"] # False
    region_attn = cra_dict["region_attn_for_cnn"] # False
    fea_size = center_attn_dicts[0]["fea_size"]
    if attn_type is None:
        assert init_state != "zero"  # img feature must be the input of decoder
    add_linear = (init_state != "zero")  # False if "zero"
    vocab_size = nc + 3  # labels + <end> + <pad> + <start>
    num_label = nc + 1  # labels + <end>
    encoder = CNN_Encoder(img_shape, backbone_name, hidden_dim,
                          extra_layer_name, ws, add_linear, region_attn)
    decoder = RNN_Decoder(fea_size, hidden_dim, vocab_size,
                          num_label, attn_type, rnn_type, 
                          center_attn_dicts, add_layer_norm)
    return encoder, decoder
