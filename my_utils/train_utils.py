import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import math
import os
import logging
from functools import partial
from .custom_metric import recall_m, recall_i, precision_m, F1Score
from .beam_search import BeamSearch
from .loss import order_the_targets_pla

class WarmUpLearningRateScheduler(tf.keras.callbacks.Callback):
    """Warmup learning rate scheduler
    """

    def __init__(self, warmup_batches, init_lr, verbose=0):
        """Constructor for warmup learning rate scheduler

        Arguments:
            warmup_batches {int} -- Number of batch for warmup.
            init_lr {float} -- Learning rate after warmup.

        Keyword Arguments:
            verbose {int} -- 0: quiet, 1: update messages. (default: {0})
        """

        super(WarmUpLearningRateScheduler, self).__init__()
        self.warmup_batches = warmup_batches
        self.init_lr = init_lr
        self.verbose = verbose
        self.batch_count = 0
        # self.learning_rates = []

    def on_batch_end(self, batch, logs=None):
        self.batch_count = self.batch_count + 1
        # lr = K.get_value(self.model.optimizer.lr)
        # self.learning_rates.append(lr)

    def on_batch_begin(self, batch, logs=None):
        if self.batch_count <= self.warmup_batches:
            lr = self.batch_count*self.init_lr/self.warmup_batches
            K.set_value(self.model.optimizer.lr, lr)
            if self.verbose > 0:
                print('\nBatch %05d: WarmUpLearningRateScheduler setting learning '
                      'rate to %s.' % (self.batch_count + 1, lr))


def dynamic_gpu_mem():
    config = tf.ConfigProto()
    # dynamically grow the memory used on the GPU
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    return sess


def build_loss(cfg):
    loss_name = cfg.LOSS_NAME
    alpha = cfg.FOCAL_LOSS_ALPHA
    if loss_name == "multi_category_focal_loss_v1":
        from .loss import multi_category_focal_loss_v1
        if alpha is None:
            alpha = [1.0]*len(cfg.LABEL_NAMES)
        assert isinstance(alpha, list)
        loss = multi_category_focal_loss_v1(alpha)
    elif loss_name == "multi_category_focal_loss_v2":
        from .loss import multi_category_focal_loss_v2
        alpha = 0.25 if alpha is None else alpha
        assert isinstance(alpha, float)
        loss = multi_category_focal_loss_v2(alpha)
    elif loss_name == "categorical_crossentropy":
        loss = tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=cfg.LABEL_SMOOTHING)
    elif loss_name in ["categorical_crossentropy_with_pc", "binary_crossentropy_with_pc"]:
        from .loss import CategoricalCrossentropyWithPC
        loss = CategoricalCrossentropyWithPC(
            cfg.PAIRWISE_CONFUSION_WEIGHT, cfg.LABEL_SMOOTHING, cfg.TASK_TYPE)
    elif loss_name == "binary_crossentropy":
        loss = tf.keras.losses.BinaryCrossentropy(
            from_logits=cfg.LOGITS_FOR_LOSS,
            label_smoothing=cfg.LABEL_SMOOTHING
        )
    else:
        loss = loss_name
    return loss


def complie_model(model, cfg):
    loss = build_loss(cfg)
    if cfg.TASK_TYPE == "multi_label":
        # for multi-label binary classification
        assert "binary_crossentropy" in cfg.LOSS_NAME
        assert "binary" in cfg.MONITOR_FOR_SAVE
        acc = 'binary_accuracy'
        metric_list = [acc, precision_m, recall_m, F1Score("f1_binary")]
    else:
        acc = 'categorical_accuracy'
        metric_list = [acc, recall_m, recall_i]

    model.compile(optimizer=cfg.OPTIMIZER, loss=loss,
                  metrics=metric_list)
    return model


def step_decay_scheduler(epoch, step_size=15, initial_lrate=0.002):
    """ learning rate decayed by `gamma` every `step_size` epochs """
    gamma = 0.1
    epochs_drop = float(step_size)
    lrate = initial_lrate * math.pow(gamma, math.floor((1+epoch)/epochs_drop))
    return lrate


def build_callbacks(cfg, steps_per_epoch_train=None, init_lr=None):
    """ build callback functions for training
    Args:
        checkpoint_dir: the dir for save checkpoint
        save_period: save model weight after every `save_period` epochs, 
                     -1 means disable
        monitor_for_save: quantity to monitor for saving the best model
        lr_strategy: "step_decay", "constant"
        init_lr: using cfg.INIT_LR as default
    Return:
        callbacks: List
    """
    checkpoint_dir, save_period = cfg.CKPT_DIR, cfg.CKPT_PERIOD
    ckpt_suffix = cfg.CKPT_SUFFIX  # ".h5" or ".tf"
    lr_strategy = cfg.LR_STRATEGY
    init_lr = cfg.INIT_LR if init_lr is None else init_lr
    warm_up_steps = cfg.WARM_UP_STEPS
    if lr_strategy == "step_decay":
        decay_epoch = cfg.LR_DECAY_SIZE
        scheduler = partial(step_decay_scheduler,
                            step_size=decay_epoch, initial_lrate=init_lr)
    elif lr_strategy == "constant":
        def scheduler(epoch): return init_lr
    else:
        NotImplementedError("%s isn't supported for lr_strategy" % lr_strategy)

    csv_path = os.path.join(checkpoint_dir, 'training_log.csv')
    csv_logger = tf.keras.callbacks.CSVLogger(csv_path, append=True)

    callbacks = [
        csv_logger,
        tf.keras.callbacks.LearningRateScheduler(scheduler),
        tf.keras.callbacks.ModelCheckpoint(os.path.join(checkpoint_dir, 'best_model%s' % ckpt_suffix),
                                           monitor=cfg.MONITOR_FOR_SAVE, verbose=1,
                                           save_best_only=True, mode='max',
                                           save_weights_only=True),
    ]
    if save_period != -1:
        weight_path = os.path.join(
            checkpoint_dir, "weight.{epoch:02d}-{val_loss:.2f}%s" % ckpt_suffix)
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(weight_path,
                                                            save_weights_only=True, period=save_period))
    if warm_up_steps:
        if isinstance(warm_up_steps, bool):
            warmup_steps = steps_per_epoch_train
        callbacks.append(WarmUpLearningRateScheduler(warmup_steps, init_lr))
    return callbacks


def class_weight_for_loss(class_weighted, cfg, df_train=None):
    if class_weighted:
        if isinstance(class_weighted, list):
            class_ws = class_weighted
        else:
            y_train = df_train[cfg.TASK_NAME].values
            class_ws = compute_class_weight('balanced',
                                            np.unique(y_train),
                                            y_train)
            class_ws = list(class_ws)
            if len(class_ws) < len(cfg.LABEL_NAMES):
                class_ws += [1.]*(len(cfg.LABEL_NAMES)-len(class_ws))
    else:
        class_ws = [1.]*len(cfg.LABEL_NAMES)
    return class_ws


def combine_history(init, fine):
    """ init = init+fine """
    for key in init.keys():
        init[key].extend(fine[key])
    return init


def build_logger(log_file_path, name="train_log"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s",
                                  datefmt='%Y-%m-%d %H:%M:%S')
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(log_file_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


@tf.function
def train_step(img_tensor, target, model, loss_function, optimizer):
    with tf.GradientTape() as tape:
        pred = model(img_tensor, training=True)
        loss = loss_function(target, pred)
    gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    return pred, loss


def get_center_layer_metrics(decoder):
    center_loss = decoder.metrics[0].result().numpy()
    center_mean = decoder.metrics[1].result().numpy()
    center_var = decoder.metrics[2].result().numpy()
    text = "center loss-%s, mean-%s, var-%s" % (
        center_loss, center_mean, center_var)
    return text


def reset_metrics(metrics):
    for metric in metrics:
        metric.reset_states()


def convert_label_for_rnn(label_batch, num_label):
    # the shape of label batch is (n, num_label)
    end_ind, pad_ind, start_ind = num_label, num_label+1, num_label+2
    max_len = tf.reduce_max(tf.reduce_sum(label_batch, axis=1)) + 1
    label_new = []
    for i in range(label_batch.shape[0]):
        inds = tf.where(label_batch[i])
        inds = tf.transpose(inds)
        inds = tf.concat([inds, [[end_ind]]], axis=1)  # <end>
        padding = [[0, 0], [0, max_len-inds.shape[1]]]
        inds = tf.pad(inds, padding, constant_values=pad_ind)
        label_new.append(inds)
    label_new = tf.concat(label_new, axis=0)
    return label_new


# @tf.function
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


def test_step_for_rnn_with_beam_search(img_tensor, decoder, encoder, num_label,
                                       initial_state="zero", beam_size=5,
                                       normalize_by_length=True):
    end_ind, pad_ind, start_ind = num_label, num_label+1, num_label+2
    max_length = num_label+1
    b_size = img_tensor.shape[0]
    beam_search = BeamSearch(decoder, beam_size, start_ind, end_ind,
                             max_length, normalize_by_length)

    fea_map, fea_vec = encoder(img_tensor)
    preds = []
    for i in range(b_size):
        # dec_input = tf.expand_dims([start_ind], 0)
        if initial_state == "zero":
            hidden = decoder.reset_state(batch_size=1)
        elif initial_state == "img_fea":
            hidden = decoder.reset_state(initial_state=fea_vec[i:i+1])
        hyps = beam_search(fea_map[i:i+1], hidden)
        best = hyps[0]
        if best.latest_token==end_ind:
            best.tokens.pop(-1) # remove <end>
        result = np.zeros(num_label)
        for pre_ind in best.tokens[1:]: # ignore <start>
            result[pre_ind] = 1
        preds.append(result)
    preds = tf.constant(np.array(preds))

    return preds


def train_step_for_rnn_with_pla(img_tensor, target, decoder, encoder,
                                loss_function, optimizer, num_label,
                                initial_state="zero", add_center_loss=False):
    end_ind, pad_ind, start_ind = num_label, num_label+1, num_label+2

    dec_input = tf.expand_dims([start_ind] * target.shape[0], 1)

    loss = 0
    center_loss = 0
    # count = 0
    pred_list = []
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
            pred_list.append(tf.expand_dims(predictions, 1))
            if add_center_loss:
                # TODO: compute center loss according to PLA
                mask = tf.cast(mask, dtype=decoder.losses[0].dtype)
                center_loss_i = tf.reduce_sum(decoder.losses[0] * mask)
                center_loss += center_loss_i
            # do not use teacher forcing
            dec_input = tf.expand_dims(tf.arg_max(predictions, 1), 1)
        pred_batch = tf.concat(pred_list, axis=1)
        mask = tf.math.logical_not(tf.math.equal(target, pad_ind))
        mask = tf.cast(mask, dtype=target_i.dtype)
        target = target*mask + end_ind*(1-mask)  # <pad> -> <end>
        label_length = tf.reduce_sum(mask, axis=1).numpy()
        target_new = order_the_targets_pla(pred_batch, target, label_length)
        cls_loss = loss_function(target_new, pred_batch)
        mask = tf.cast(mask, dtype=cls_loss.dtype)
        cls_loss = tf.reduce_sum(cls_loss*mask)
        loss = cls_loss + center_loss
        mean_loss = loss/tf.reduce_sum(mask)

    trainable_variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(mean_loss, trainable_variables)

    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss, mean_loss