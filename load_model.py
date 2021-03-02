import os,sys,shutil,ast,math

import pandas as pd
import numpy as np

# Helper libraries
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
# Custom libraries
from my_utils import load_data_utils as ldu
from my_utils import train_utils, eval_utils, networks
from tqdm import tqdm 


def train_step_for_rnn(img_tensor, target, decoder, encoder,
                       loss_function, optimizer, num_label,
                       initial_state="zero", add_center_loss=False):
    end_ind, pad_ind, start_ind = num_label, num_label+1, num_label+2

    dec_input = tf.expand_dims([start_ind] * target.shape[0], 1)

    loss = 0
    count = 0
    with tf.GradientTape() as tape:
        fea_map, fea_vec = encoder(img_tensor, training=True)
        # initializing the hidden state for each batch
        # because the captions are not related from image to image
        if initial_state == "zero":
            hidden = decoder.reset_state(batch_size=target.shape[0])
        elif initial_state == "img_fea":
            hidden = decoder.reset_state(initial_state=fea_vec)
        for i in range(target.shape[1]):
            # passing the features through the decoder
            target_i = target[:, i]
            mask = tf.math.logical_not(tf.math.equal(target_i, pad_ind))
            mask = tf.cast(mask, dtype=target_i.dtype)
            target_i = target_i*mask + end_ind*(1-mask)  # <pad> -> <end>
            predictions, hidden, _ = decoder(dec_input, fea_map,
                                             hidden, target_i, True)

                                             
            loss_i = loss_function(target_i, predictions)
            mask = tf.cast(mask, dtype=loss_i.dtype)
            loss_i = tf.reduce_sum(loss_i * mask)
            count += tf.reduce_sum(mask)
            loss += loss_i
            if add_center_loss:
                center_loss_i = tf.reduce_sum(decoder.losses[0] * mask)
                loss += center_loss_i
            # using teacher forcing
            dec_input = tf.expand_dims(target[:, i], 1)
        mean_loss = (loss / count)

    trainable_variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(mean_loss, trainable_variables)

    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss, mean_loss


def eval_step_for_rnn(img_tensor, decoder, encoder, num_label, initial_state="zero"):
    end_ind, pad_ind, start_ind = num_label, num_label+1, num_label+2
    max_length = num_label+1
    b_size = img_tensor.shape[0]

    fea_map, fea_vec = encoder(img_tensor)
    preds = []
    for i in range(b_size):
        dec_input = tf.expand_dims([start_ind], 0)
        if initial_state == "zero":
            hidden = decoder.reset_state(batch_size=1)
        elif initial_state == "img_fea":
            hidden = decoder.reset_state(initial_state=fea_vec[i:i+1])
        result = np.zeros(num_label)
        for j in range(max_length):
            predictions, hidden, attn_weights = decoder(
                dec_input, fea_map[i:i+1], hidden)
            pred_ind = tf.argmax(predictions, 1)
            pred_ind_scale = pred_ind.numpy()[0]
            if pred_ind_scale == end_ind:
                break
            result[pred_ind_scale] = 1
            dec_input = tf.expand_dims(pred_ind, 0)
        preds.append(result)
    preds = tf.constant(np.array(preds))

    return preds


def main(cfg, skip_train=False, distributed_train=False):
    # assert cfg.EAGER_EXECUTION == True, "only support eager mode"
    tf.enable_eager_execution()
    assert cfg.LOGITS_FOR_LOSS == True
    init_hidden_state = cfg.CRA_INIT_HIDDEN_STATE  # "zero" or "img_fea"
    add_center_loss = cfg.CRA_CA_ADD_CENTER_LAYER
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=cfg.LOGITS_FOR_LOSS, reduction='none')
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    
    encoder, decoder = networks.build_model(cfg)
    ckpt = tf.train.Checkpoint(encoder=encoder,
                                decoder=decoder,
                                optimizer=optimizer)
    





if __name__ == "__main__":
    from argparse import ArgumentParser
    from my_utils.default_config import dict2config, update_and_save_cfg, CONFIG_DICT
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    import tensorflow as tf
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    

    config_paths = [
                    './configs/tianchi_resnet34_gru_spatial_hidden_concat_center_attn5part_multi_head_conv4_rgb192x480.json'
                    ]

    for config_path in config_paths:
        st = True
        CONFIG_DICT["TMP_NUM_GPUS"] = 1
        cfg_dict = update_and_save_cfg(CONFIG_DICT, config_path,
                                       save_cfg=not st, date=not st)
        cfg = dict2config(cfg_dict)
        main(cfg)
