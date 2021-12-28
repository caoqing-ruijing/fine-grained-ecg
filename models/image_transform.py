import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K


# borrowed from https://stackoverflow.com/questions/34801342/tensorflow-how-to-rotate-an-image-for-data-augmentation
@tf.function
def rotate_image_tensor(image, angle, size, mode='white'):
    """
    Rotates a 3D tensor (HWD), which represents an image by given radian angle.

    New image has the same size as the input image.

    mode controls what happens to border pixels.
    mode = 'black' results in black bars (value 0 in unknown areas)
    mode = 'white' results in value 255 in unknown areas
    mode = 'ones' results in value 1 in unknown areas
    mode = 'repeat' keeps repeating the closest pixel known
    mode is int/float, filled with the specified value
    """
    s = size  # image.get_shape().as_list()
    assert len(s) == 3, "Input needs to be 3D."
    assert (mode == 'repeat') or (mode == 'black') or (mode == 'white') or (mode == 'ones') \
        or isinstance(mode, int) or isinstance(mode, float),  "Unknown boundary mode."
    image_center = (s[0]//2, s[1]//2)

    # Coordinates of new image
    coord1 = tf.range(s[0])
    coord2 = tf.range(s[1])

    # Create vectors of those coordinates in order to vectorize the image
    coord1_vec = tf.tile(coord1, [s[1]])

    coord2_vec_unordered = tf.tile(coord2, [s[0]])
    coord2_vec_unordered = tf.reshape(coord2_vec_unordered, [s[0], s[1]])
    coord2_vec = tf.reshape(tf.transpose(coord2_vec_unordered, [1, 0]), [-1])

    # center coordinates since rotation center is supposed to be in the image center
    coord1_vec_centered = coord1_vec - image_center[0]
    coord2_vec_centered = coord2_vec - image_center[1]

    coord_new_centered = tf.cast(
        tf.stack([coord1_vec_centered, coord2_vec_centered]), tf.float32)

    # Perform backward transformation of the image coordinates
    rot_mat_inv = tf.dynamic_stitch([[0], [1], [2], [3]], [tf.cos(
        angle), tf.sin(angle), -tf.sin(angle), tf.cos(angle)])
    rot_mat_inv = tf.reshape(rot_mat_inv, shape=[2, 2])
    coord_old_centered = tf.matmul(rot_mat_inv, coord_new_centered)

    # Find nearest neighbor in old image
    coord1_old_nn = tf.cast(
        tf.round(coord_old_centered[0, :] + image_center[0]), tf.int32)
    coord2_old_nn = tf.cast(
        tf.round(coord_old_centered[1, :] + image_center[1]), tf.int32)

    # Clip values to stay inside image coordinates
    if mode == 'repeat':
        coord_old1_clipped = tf.minimum(tf.maximum(coord1_old_nn, 0), s[0]-1)
        coord_old2_clipped = tf.minimum(tf.maximum(coord2_old_nn, 0), s[1]-1)
    else:
        outside_ind1 = tf.logical_or(tf.greater(
            coord1_old_nn, s[0]-1), tf.less(coord1_old_nn, 0))
        outside_ind2 = tf.logical_or(tf.greater(
            coord2_old_nn, s[1]-1), tf.less(coord2_old_nn, 0))
        outside_ind = tf.logical_or(outside_ind1, outside_ind2)

        coord_old1_clipped = tf.boolean_mask(
            coord1_old_nn, tf.logical_not(outside_ind))
        coord_old2_clipped = tf.boolean_mask(
            coord2_old_nn, tf.logical_not(outside_ind))

        coord1_vec = tf.boolean_mask(coord1_vec, tf.logical_not(outside_ind))
        coord2_vec = tf.boolean_mask(coord2_vec, tf.logical_not(outside_ind))

    coord_old_clipped = tf.cast(tf.transpose(
        tf.stack([coord_old1_clipped, coord_old2_clipped]), [1, 0]), tf.int32)

    # Coordinates of the new image
    coord_new = tf.transpose(
        tf.cast(tf.stack([coord1_vec, coord2_vec]), tf.int32), [1, 0])

    image_channel_list = tf.split(image, s[2], 2)

    image_rotated_channel_list = list()
    for image_channel in image_channel_list:
        image_chan_new_values = tf.gather_nd(
            tf.squeeze(image_channel), coord_old_clipped)

        if (mode == 'black') or (mode == 'repeat'):
            background_color = 0
        elif mode == 'ones':
            background_color = 1
        elif mode == 'white':
            background_color = 255
        else:
            background_color = mode

        image_rotated_channel_list.append(tf.sparse_to_dense(coord_new, [s[0], s[1]], image_chan_new_values,
                                                             background_color, validate_indices=False))

    image_rotated = tf.transpose(
        tf.stack(image_rotated_channel_list), [1, 2, 0])

    return image_rotated


# refer to https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py#L93-L149
def preprocess_symbolic_input(x, data_format="channels_last", mode="tf", imagenet_norm=True):
    """Preprocesses a tensor encoding a batch of images.
    # Arguments
        x: Input tensor, 3D or 4D.
        data_format: Data format of the image tensor.
        mode: One of "caffe", "tf" or "torch".
            - caffe: will convert the images from RGB to BGR,
                then will zero-center each color channel with
                respect to the ImageNet dataset,
                without scaling.
            - tf: will scale pixels between -1 and 1,
                sample-wise.
            - torch: will scale pixels between 0 and 1 and then
                will normalize each channel with respect to the
                ImageNet dataset.
    # Returns
        Preprocessed tensor.
    """

    if mode == 'tf':
        x /= 127.5
        x -= 1.
        return x

    if mode == 'torch':
        x /= 255.
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        if data_format == 'channels_first':
            # 'RGB'->'BGR'
            if K.ndim(x) == 3:
                x = x[::-1, ...]
            else:
                x = x[:, ::-1, ...]
        else:
            # 'RGB'->'BGR'
            x = x[..., ::-1]
        x = K.cast(x, K.floatx())
        mean = [103.939, 116.779, 123.68]
        std = None
    
    if imagenet_norm:
        mean_tensor = K.constant(-np.array(mean))

        # Zero-center by mean pixel
        if K.dtype(x) != K.dtype(mean_tensor):
            x = K.bias_add(
                x, K.cast(mean_tensor, K.dtype(x)),
                data_format=data_format)
        else:
            x = K.bias_add(x, mean_tensor, data_format)
        if std is not None:
            x /= std
    return x


# ------------------------------ data augmentation for 1D signal ------------------------------

def random_crop_and_resize_1d(x, length, max_side_crop=0.01):
    """ random crop the input, and then resize to the original size
    x:  the input Tensor with size = (length, dim)
    length = the length of the input
    max_side_crop: the factor for cropping
    """
    random_range = tf.random.uniform(shape=(2,), minval=0, maxval=int(length*max_side_crop))
    random_range = tf.cast(tf.round(random_range), tf.int32)
    x_crop = x[None, random_range[0]:length-random_range[1], :]
    x = tf.image.resize(x_crop, (1, length)) # (1, length, dim)
    return x[0]


def random_shift_1d(signal, len_dim, wrg=0.01, pad_value=0.0):
    """ white padding, then random crop 
    signal:  the input Tensor with size = (length, dim)
    len_dim: the shape of the input
    wrg: the factor for shifting
    pad_value: the value for padding
    """
    length, dim = len_dim
    pad_w = int(length*wrg)
    paddings = tf.constant([[pad_w, pad_w], [0, 0]])
    signal_pad = tf.pad(signal, paddings, "CONSTANT", constant_values=pad_value)
    signal_crop = tf.image.random_crop(signal_pad[None,:,:], (1, length, dim)) # (1, length, dim)
    return signal_crop[0]


def process_each_lead_in_4x10s(rows, fn1, fn2=None):
    if fn2 is None:
        fn2 = fn1 # for the last row
    row_list = tf.split(rows, 4, axis=1) # (5000, 4) -> [(5000,1), (5000,1), (5000,1), (5000,1)]
    row_list_new = []
    for i in range(3):
        lead_list_i = tf.split(row_list[i], 4, axis=0)
        lead_list_i = [fn1(x) for x in lead_list_i]
        row_list_new.append(tf.concat(lead_list_i, axis=0))
    row = fn2(row_list[-1])
    row_list_new.append(row)
    rows_new = tf.concat(row_list_new, axis=1)
    return rows_new