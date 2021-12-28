import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

from .networks import build_backbone, build_final_layer
from .fine_grained_net import CenterLayer, norm_fea, flatten_hw


def call_layers(layers, x, training=False):
    for layer in layers:
        x = layer(x, training=training)
    return x


class CenterAttentionV2(tf.keras.Model):
    def __init__(self, hidden_dim, num_class, center_attn_dict, multi_head_attn_dict):
        super(CenterAttentionV2, self).__init__()
        dropout_rate = center_attn_dict["dropout_rate"]  # 0.5
        self.part_num = center_attn_dict["part_num"]
        self.conv_bn_relu_list = []
        self.pool_bn_relu_dropout_list = []
        self.region_bn_list = []
        for _ in range(self.part_num+1):
            tmp = [keras.layers.Conv2D(hidden_dim, (1, 1), padding='same'),
                   keras.layers.BatchNormalization(epsilon=1.001e-5),
                   keras.layers.Activation('relu')]
            self.conv_bn_relu_list.append(tmp)

            tmp = [keras.layers.GlobalAveragePooling2D(),
                   keras.layers.BatchNormalization(epsilon=1.001e-5),
                   keras.layers.Activation('relu'),
                   keras.layers.Dropout(dropout_rate)]
            self.pool_bn_relu_dropout_list.append(tmp)

            self.region_bn_list.append(tf.keras.layers.BatchNormalization(epsilon=1.001e-5))

        self.hidden_add = keras.layers.Add()
        self.sigmoid = tf.keras.layers.Activation('sigmoid')

        add_center_layer = center_attn_dict["add_center_layer"]
        if add_center_layer:
            fea_dim = center_attn_dict["fea_dim"]
            beta_c, lambda_c = center_attn_dict["beta_c"], center_attn_dict["lambda_c"]
            self.center_layer = CenterLayer(num_class, self.part_num, fea_dim,
                                            beta_c, lambda_c, "multi_class", False)
        else:
            self.center_layer = None

        add_multi_head_attn = multi_head_attn_dict["add_multi_head_attn"]
        if add_multi_head_attn:
            assert add_center_layer == True
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
            if self.add_norm_dense_after_attn:
                self.multi_head_norm1 = keras.layers.LayerNormalization(epsilon=1e-6)
                self.multi_head_ffn = multi_head_attn.point_wise_feed_forward_network(
                    fea_dim, max(128, fea_dim//4))
            self.multi_head_add = keras.layers.Add()
            self.multi_head_norm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        else:
            self.multi_head_attn_layer = None

    def call(self, fea_map, hidden, label=None, training=False):
        hidden = tf.expand_dims(tf.expand_dims(hidden, 1), 1)  # (n, 1, 1, hidden_dim)
        cls_fea_map_ori = call_layers(self.conv_bn_relu_list[-1], fea_map, training)
        cls_fea_map, HxW = flatten_hw(cls_fea_map_ori)

        attr_pool_list = []
        for i in range(self.part_num):
            # (n, H, W, hidden_dim)
            attr_fea_map_i = call_layers(self.conv_bn_relu_list[i], fea_map, training)
            attr_fea_map_i = self.hidden_add([attr_fea_map_i, hidden]) # attr_pool_i += hidden
            attr_pool_i = call_layers(self.pool_bn_relu_dropout_list[i], attr_fea_map_i, training)
            attr_pool_i = tf.expand_dims(attr_pool_i, -1)  # (n, hidden_dim, 1)
            attr_pool_list.append(attr_pool_i)

        # TODO: `fea_map` -> `cls_fea_map`
        fea_map_ = K.permute_dimensions(cls_fea_map, (0, 2, 1))  # (n, hidden_dim, HxW)
        # fea_map_, HxW = flatten_hw(fea_map)   # (n, HxW, fea_dim)
        # fea_map_ = K.permute_dimensions(fea_map_, (0, 2, 1))  # (n, fea_dim, HxW)
        region_attn_map_list = []
        part_list = []
        for i in range(self.part_num):
            attn_cls = K.batch_dot(cls_fea_map, attr_pool_list[i])  # (n, HxW, 1)
            region_attn_map = self.sigmoid(attn_cls)
            region_attn_map /= tf.cast(HxW, tf.float32)
            region_fea = K.batch_dot(fea_map_, region_attn_map)  # (n, hidden_dim, 1)
            region_fea = tf.squeeze(region_fea, -1)  # (n, hidden_dim)
            region_fea = self.region_bn_list[i](region_fea, training=training)
            region_fea = tf.expand_dims(region_fea, 1) # (n, 1, hidden_dim)
            part_list.append(region_fea)
            region_attn_map_list.append(region_attn_map)

        if len(part_list) > 1:
            attn_maps = keras.layers.concatenate(region_attn_map_list, axis=2) # (n, HxW, m)
            bap = keras.layers.concatenate(part_list, axis=1)  # (n, m, hidden_dim)
        else:
            attn_maps = region_attn_map_list[0]
            bap = part_list[0]
        if self.center_layer is not None:
            bap = self.center_layer((bap, label), training=training)

        if self.multi_head_attn_layer is not None:
            bap_attn, _ = self.multi_head_attn_layer(bap, k=bap, q=bap)
            if self.multi_head_dropout is not None:
                bap_attn = self.multi_head_dropout(
                    bap_attn, training=training)
            bap = self.multi_head_add([bap, bap_attn])  # (n, m, hidden_dim)
            if self.add_norm_dense_after_attn:
                bap = self.multi_head_norm1(bap, training=training)
                # (batch_size, input_seq_len, d_model)
                ffn_output = self.multi_head_ffn(bap)
                if self.multi_head_dropout is not None:
                    ffn_output = self.multi_head_dropout(
                        ffn_output, training=training)
                bap = self.multi_head_add([bap, ffn_output])
                bap = self.multi_head_norm2(bap, training=training)
        bap = keras.layers.Flatten()(bap)  # (n, mxhidden_dim)
        return bap, attn_maps
        

class CenterAttention(tf.keras.Model):
    def __init__(self, hidden_dim, num_class, center_attn_dict, multi_head_attn_dict):
        super(CenterAttention, self).__init__()
        self.part_num = center_attn_dict["part_num"]
        self.fea_conv = keras.layers.Conv2D(hidden_dim, (1, 1), strides=(1, 1), padding='same',
                                            kernel_initializer='glorot_uniform')
        self.attn_conv = keras.layers.Conv2D(self.part_num, (1, 1), strides=(1, 1), padding='same',
                                             kernel_initializer='glorot_uniform')
        dropout_rate = center_attn_dict["dropout_rate"]  # 0.2
        self.attn_dropout = keras.layers.SpatialDropout2D(dropout_rate)
        self.pool_layer = keras.layers.GlobalAveragePooling2D()

        fea_dim = center_attn_dict["fea_dim"]
        gate_for_bap = center_attn_dict["gate_for_bap"]
        if gate_for_bap:
            self.gate_fc = keras.layers.Dense(fea_dim)
        else:
            self.gate_fc = None
        self.use_spatial_hidden = center_attn_dict["use_spatial_hidden"] # TODO: fix error for beam search
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
            if self.add_norm_dense_after_attn:
                self.multi_head_norm1 = keras.layers.LayerNormalization(epsilon=1e-6)
                self.multi_head_ffn = multi_head_attn.point_wise_feed_forward_network(
                    fea_dim, max(128, fea_dim//4))
            self.multi_head_add = keras.layers.Add()
            self.multi_head_norm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        else:
            self.multi_head_attn_layer = None

    def call(self, fea_map, hidden, label=None, training=False):
        if self.use_spatial_hidden:
            hidden_ = self.hidden_spatial_fc(hidden)
            hidden_ = tf.reshape(hidden_, (-1, self.fea_h, self.fea_w, 1))
            if self.fuse_hidden_method == "add":
                fea_map_hidden = self.fea_conv(fea_map) + hidden_
            else:
                fea_map_hidden = tf.concat([self.fea_conv(fea_map), hidden_], axis=-1)
        else:
            hidden_ = tf.expand_dims(tf.expand_dims(
                hidden, 1), 1)  # (n, 1, 1, hidden_dim)
            fea_map_hidden = self.fea_conv(fea_map) + hidden_
        attn_maps = self.attn_conv(fea_map_hidden)
        attn_maps = self.attn_dropout(attn_maps, training=training)
        attn_maps = tf.split(attn_maps, self.part_num,
                             axis=-1)  # [a_1, a_2, ... , a_m]
        if self.gate_fc is not None:
            gate = tf.math.sigmoid(self.gate_fc(hidden))
        else:
            gate = None

        part_list = []
        for ai in attn_maps:
            part = fea_map * ai  # (n, H, W, C)
            part = self.pool_layer(part)  # (n, C)
            if gate is not None:
                part *= gate
            part = tf.expand_dims(part, 1)  # (n, 1, C)
            part_list.append(part)
        if len(part_list) > 1:
            bap = keras.layers.concatenate(part_list, axis=1)  # (n, m, C)
        else:
            bap = part_list[0]
        if self.center_layer is not None:
            bap = self.center_layer((bap, label), training=training)
        bap = norm_fea(bap)
        if self.multi_head_attn_layer is not None:
            bap_attn, _ = self.multi_head_attn_layer(bap, k=bap, q=bap)
            if self.multi_head_dropout is not None:
                bap_attn = self.multi_head_dropout(
                    bap_attn, training=training)
            bap = self.multi_head_add([bap, bap_attn])  # (n, m, C)
            if self.add_norm_dense_after_attn:
                bap = self.multi_head_norm1(bap, training=training)
                # (batch_size, input_seq_len, d_model)
                ffn_output = self.multi_head_ffn(bap)
                if self.multi_head_dropout is not None:
                    ffn_output = self.multi_head_dropout(
                        ffn_output, training=training)
                bap = self.multi_head_add([bap, ffn_output])
                bap = self.multi_head_norm2(bap, training=training)
        bap = keras.layers.Flatten()(bap)  # (n, mxC)
        return bap, attn_maps


# borrowed from https://tensorflow.google.cn/tutorials/text/image_captioning
class BahdanauAttention(tf.keras.Model):
    def __init__(self, units, gate=False):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
        if gate:
            self.gate_fc = tf.keras.layers.Dense(units)
        else:
            self.gate_fc = None

    def call(self, features, hidden):
        b_size = tf.shape(features)[0]
        embedding_dim = features.shape[-1]
        features = tf.reshape(features, (b_size, -1, embedding_dim))
        # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # score shape == (batch_size, 64, hidden_size)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

        # attention_weights shape == (batch_size, 64, 1)
        # you get 1 at the last axis because you are applying score to self.V
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        if self.gate_fc is not None:
            gate = tf.math.sigmoid(self.gate_fc(hidden))
            context_vector *= gate

        return context_vector, attention_weights


def build_backbone_with_2outs(backbone_name, input_shape,
                              extra_layer_name=None,
                              weights="imagenet"):
    backbone = build_backbone(backbone_name, input_shape,
                              weights=weights, develop_dict=None)
    if extra_layer_name is not None:
        target_layer = backbone.get_layer(extra_layer_name)
    else:
        target_layer = backbone
    img = backbone.input
    mid_map = target_layer.output
    final_map = backbone.output
    model = tf.keras.Model(inputs=img, 
                           outputs=[mid_map, final_map],
                           name="cnn_with_two_outs")
    return model


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
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
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
        if self.region_attn:
            cls_fea_map_ori = call_layers(self.conv_bn_relu_list[1], fea_vec, training)
            cls_fea_map, HxW = flatten_hw(cls_fea_map_ori)

            attr_fea_map_i = call_layers(self.conv_bn_relu_list[0], fea_vec, training)
            attr_pool_i = call_layers(self.pool_bn_relu_dropout, attr_fea_map_i, training)
            attr_pool_i = tf.expand_dims(attr_pool_i, -1)  # (n, hidden_dim, 1)

            # TODO: `fea_map` -> `cls_fea_map`
            fea_map_ = K.permute_dimensions(cls_fea_map, (0, 2, 1))  # (n, hidden_dim, HxW)
            # fea_map_, HxW = flatten_hw(fea_vec)   # (n, HxW, fea_dim)
            # fea_map_ = K.permute_dimensions(fea_map_, (0, 2, 1))  # (n, fea_dim, HxW)
            attn_cls = K.batch_dot(cls_fea_map, attr_pool_i)  # (n, HxW, 1)
            region_attn_map = self.sigmoid(attn_cls)
            region_attn_map /= tf.cast(HxW, tf.float32)
            region_fea = K.batch_dot(fea_map_, region_attn_map)  # (n, hidden_dim, 1)
            fea_vec = tf.squeeze(region_fea, -1)  # (n, hidden_dim)
            fea_vec = self.region_bn(fea_vec, training=training)
        else:
            if self.add_linear:
                fea_vec = self.fc(fea_vec)  # (n, embedding_dim)
                fea_vec = tf.nn.relu(fea_vec)
            fea_vec = self.pool(fea_vec)
        
        return fea_map, fea_vec


class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_label,
                 attn_type="Bahdanau", rnn_type="gru", 
                 center_attn_dicts=None, add_layer_norm=False):
        super(RNN_Decoder, self).__init__()
        self.units = hidden_dim
        self.rnn_type = rnn_type

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        if self.rnn_type == "gru":
            self.rnn = tf.keras.layers.GRU(
                self.units, return_sequences=True, return_state=True,
                recurrent_initializer='glorot_uniform'
            )
        elif self.rnn_type == "lstm":
            self.rnn = tf.keras.layers.LSTM(
                self.units, return_sequences=True, return_state=True,
                recurrent_initializer='glorot_uniform'
            )
        # self.fc1 = tf.keras.layers.Dense(self.units)
        if add_layer_norm:
            self.norm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        else:
            self.norm2 = None
        self.fc2 = tf.keras.layers.Dense(num_label)

        self.attn_type = attn_type
        if self.attn_type == "Bahdanau":
            self.attention = BahdanauAttention(self.units) # TODO: gate=True
        elif self.attn_type == "center_attn":
            self.attention = CenterAttention(hidden_dim, num_label,
                                             center_attn_dicts[0], 
                                             center_attn_dicts[1])
        elif self.attn_type == "center_attn_v2":
            self.attention = CenterAttentionV2(hidden_dim, num_label,
                                               center_attn_dicts[0], 
                                               center_attn_dicts[1])
        else:
            self.attention = None

    def call(self, x, features, hidden, label=None, training=False):
        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        if self.attention is not None:
            # defining attention as a separate model
            if "center_attn" in self.attn_type:
                context_vec, attn_weights = self.attention(
                    features, hidden[0], label, training)
            else:
                context_vec, attn_weights = self.attention(features, hidden[0])

            # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
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
        x = tf.squeeze(output, 1)
        if self.norm2 is not None:
            x = self.norm2(x, training=training)

        # output shape == (batch_size * max_length, vocab)
        x = self.fc2(x)

        return x, state, attn_weights

    def reset_state(self, batch_size=1, initial_state='zeros'):
        if initial_state == 'zeros':
            initial_state = tf.zeros((batch_size, self.units))

        if self.rnn_type == "gru":
            return (initial_state,)
        elif self.rnn_type == "lstm":
            return (initial_state, initial_state)


def build_cnn_rnn_with_attn(img_shape, ws, nc, cra_dict, center_attn_dicts=None):
    backbone_name = cra_dict["backbone_name"]
    extra_layer_name = cra_dict["extra_layer_name"]  # None
    rnn_type = cra_dict["rnn_type"]  # "gru" or "lstm"
    attn_type = cra_dict["attn_type"]  # "Bahdanau", "center_attn", "center_attn_v2"
    embedding_dim = cra_dict["embedding_dim"]
    hidden_dim = cra_dict["hidden_dim"]  # 512
    init_state = cra_dict["init_hidden_state"]  # "zero" or "img_fea"
    add_layer_norm = cra_dict["add_layer_norm_after_rnn"] # False
    region_attn = cra_dict["region_attn_for_cnn"] # False
    if attn_type is None:
        assert init_state != "zero"  # img feature must be the input of decoder
    add_linear = (init_state != "zero")  # False if "zero"
    vocab_size = nc + 3  # labels + <end> + <pad> + <start>
    num_label = nc + 1  # labels + <end>
    encoder = CNN_Encoder(img_shape, backbone_name, hidden_dim,
                          extra_layer_name, ws, add_linear, region_attn)
    decoder = RNN_Decoder(embedding_dim, hidden_dim, vocab_size,
                          num_label, attn_type, rnn_type, 
                          center_attn_dicts, add_layer_norm)
    return encoder, decoder


def cnn_lstm(input_shape, num_class,
             freeze_backbone=False, weights='imagenet',
             initial_bias=None, final_activation="sigmoid",
             cnn_lstm_dict=None, develop_dict=None):
    backbone_name = cnn_lstm_dict["backbone_name"]  # "develop_net"
    extra_layer_name = cnn_lstm_dict["extra_layer_name"]  # "conv4_block2_out"
    backbone = build_backbone(backbone_name, input_shape,
                              weights=weights, develop_dict=develop_dict)
    if freeze_backbone:
        backbone.trainable = False
    if extra_layer_name is not None:
        target_layer = backbone.get_layer(extra_layer_name)
    else:
        target_layer = backbone
    img = backbone.input
    fea_map = target_layer.output
    b_size = tf.shape(fea_map)[0]
    f_shape = target_layer.output_shape  # (n, H, W, C)
    fea_map = tf.transpose(fea_map, [0, 2, 1, 3])  # (n, W, H, C)
    target_shape = (b_size, f_shape[2], f_shape[1]*f_shape[3])
    fea_map = tf.reshape(fea_map, target_shape)
    hhc = keras.layers.CuDNNLSTM(
        f_shape[3], return_state=True)(fea_map)  # (n, 512)
    x = hhc[0]
    x = build_final_layer(num_class, initial_bias)(x)
    prob = keras.layers.Activation(final_activation)(x)

    model = tf.keras.Model(inputs=img, outputs=prob,
                           name="cnn_lstm")
    return model


class RNN_Module(keras.layers.Layer):
    """ RNN Module """

    def __init__(self, class_num, hidden_dim, label_embed_dim,
                 final_activation="sigmoid", **kwargs):
        super().__init__(**kwargs)
        self.class_num = class_num
        self.hidden_dim = hidden_dim
        self.final_activation = final_activation
        self.label_embed_dim = label_embed_dim

    def build(self, input_shape):
        self.rnn = keras.layers.CuDNNLSTM(
            self.hidden_dim, return_sequences=True, return_state=True)
        self.label_embed_mat = keras.layers.Dense(self.label_embed_dim)
        self.dense = keras.layers.Dense(1, use_bias=False)
        self.final_activation = keras.layers.Activation(self.final_activation)

    def call(self, x, training=None):
        fea_map, label = x[0], x[1]
        b_size = tf.shape(fea_map)[0]
        if training is None:
            training = K.learning_phase()
        if True:
            # if training:
            #     paddings = tf.constant([[0, 0,], [1, 0]])
            #     label_shift = tf.pad(label, paddings)[:,:-1]
            #     label_shift = tf.expand_dims(label_shift, axis=-1) # (n, T, 1)
            #     label_shift = tf.eye(self.class_num, batch_shape=(1,)) * label_shift # (n, T, T)
            #     label_embed = self.label_embed_mat(label_shift) # (n, T, m)
            #     # the shape of `rnn_output` is [(n, T, hidden_dim), (n, hidden_dim), (n, hidden_dim)]
            #     rnn_output = self.rnn(label_embed, [fea_map, fea_map])
            #     x = self.dense(rnn_output[0]) # (n, T, 1)
            #     x = tf.squeeze(x, axis=-1)
            #     probs = self.final_activation(x) # (n, T)
            # else:
            label_pre = tf.zeros([b_size, 1, self.class_num])  # (n, 1, m)
            label_pre_embed = self.label_embed_mat(label_pre)
            states = [fea_map, fea_map]
            probs = []
            for i in range(self.class_num):
                rnn_output = self.rnn(label_pre_embed, states)
                x = self.dense(rnn_output[0])  # (n, 1, 1)
                prob = self.final_activation(x)  # (n, 1, 1)
                label_pre = tf.eye(self.class_num, batch_shape=(1,))[
                    :, i:i+1, :]
                label_pre *= prob  # (n, 1, T)
                label_pre_embed = self.label_embed_mat(label_pre)  # (n, 1, m)
                states = rnn_output[1:]
                probs.append(prob)
            probs = tf.squeeze(tf.concat(probs, axis=1), axis=-1)

        return probs

    def get_config(self):
        config = super().get_config()
        config.update({"class_num": self.class_num,
                       "hidden_dim": self.hidden_dim,
                       "final_activation": self.final_activation})


def cnn_rnn(input_shape, num_class,
            freeze_backbone=False, weights='imagenet',
            initial_bias=None, final_activation="sigmoid",
            cnn_rnn_dict=None, develop_dict=None):
    backbone_name = cnn_rnn_dict["backbone_name"]  # "develop_net"
    extra_layer_name = cnn_rnn_dict["extra_layer_name"]  # "conv4_block2_out"
    label_embed_dim = 8
    pool_layer = keras.layers.GlobalAveragePooling2D()
    backbone = build_backbone(backbone_name, input_shape,
                              weights=weights, develop_dict=develop_dict)
    if freeze_backbone:
        backbone.trainable = False
    if extra_layer_name is not None:
        target_layer = backbone.get_layer(extra_layer_name)
    else:
        target_layer = backbone
    img = backbone.input
    label = keras.layers.Input(
        (num_class,), dtype=tf.float32, name="input_label")
    fea_map = target_layer.output
    fea_map_shape = target_layer.output_shape  # (n, H, W, C)
    fea_dim = fea_map_shape[-1]  # as the hidden dim of RNN

    fea_map = pool_layer(fea_map)  # (n, C)
    rnn_module = RNN_Module(
        num_class, fea_dim, label_embed_dim, final_activation)
    prob = rnn_module([fea_map, label])

    model = tf.keras.Model(inputs=[img, label], outputs=prob,
                           name="cnn_rnn")
    return model
