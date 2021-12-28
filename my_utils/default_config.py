from collections import namedtuple
import json
import time
import os
import copy
from .dataset_info import LABEL_NAME_DICT


def dict2config(config_dict):
    """ convert dict to config """
    cfg_names = config_dict.keys()
    Config = namedtuple("Config", cfg_names)
    cfg = Config(*[config_dict[key] for key in cfg_names])
    return cfg


def update_and_save_cfg(default_dict, json_path=None, save_cfg=True, date=None):
    """ update dict according to the json file
    Args:
        default_dict: config dict
        json_path: the path of json file ("xxx.json")
        save_cfg: whether saving config as json file
        date:add date at the end of the "CKPT_DIR",
        True means using today, `str` to specify the date, False means unchange
    """
    # update config
    default_dict = copy.deepcopy(default_dict)
    if json_path is not None:
        with open(json_path, "r") as f:
            new_dict = json.load(f)
        default_dict.update(new_dict)
    if "CKPT_DIR" not in new_dict.keys():
        json_name = os.path.basename(json_path)[:-5]
        default_dict["CKPT_DIR"] = os.path.join("checkpoint", json_name)
    # the date of "CKPT_DIR"
    if date:
        if not isinstance(date, str):
            date = time.strftime("%2m%2d", time.localtime())  # month-day
        default_dict["CKPT_DIR"] = default_dict["CKPT_DIR"] + '_' + date
    # label names
    if "LABEL_NAMES" not in new_dict.keys():
        csv_name = os.path.basename(default_dict["CSV_PATH"])
        default_dict["LABEL_NAMES"] = LABEL_NAME_DICT[csv_name][default_dict["TASK_NAME"]]
    if default_dict["TWO_STAGE_FINETUNE"]:
        assert default_dict["FREEZE_BACKBONE"] == True
    # save config
    if save_cfg:
        # delete tmp configs which starts with "TMP_"
        key_list = list(default_dict.keys())
        default_dict_for_save = copy.deepcopy(default_dict)
        for key in key_list:
            if key.startswith("TMP_"):
                default_dict_for_save.pop(key)
        # save in `CKPT_DIR`
        save_dir = default_dict_for_save["CKPT_DIR"]
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "train_config.json"), "w") as f:
            json.dump(default_dict_for_save, f, indent=4)
    return default_dict


# dict of default config
CONFIG_DICT = {}

####### dataset #######
CONFIG_DICT["CSV_PATH"] = 'datasets/save_data_profile.csv'
CONFIG_DICT["IMG_ROOT"] = "datasets/ST_Exclude_bottom_denoise"
CONFIG_DICT["IMG_SUFFIX"] = ".jpg"
# opts: "*/*"(image folders by class), "*"(all images in one folder)
CONFIG_DICT["GLOB_PATTERN"] = "*"
# opts: "Label_Noise", "Label_rhythm"
CONFIG_DICT["TASK_NAME"] = "Label_Noise"
CONFIG_DICT["TASK_TYPE"] = "multi_class" # or "multi_label"
CONFIG_DICT["FEA_NAMES"] = ['Gender', 'Age',
                            'HR', 'PR', 'QT', 'QTc', 'QRS', 'QRSE']
# automatic loading according to `dataset_info.py`
CONFIG_DICT["LABEL_NAMES"] = None
# opts: "img", "img_fea"
CONFIG_DICT["INPUT_TYPE"] = "img"
CONFIG_DICT["DIGITAL_TRANS_TYPE"] = "4*10s"
CONFIG_DICT["IMG_HEIGHT"] = CONFIG_DICT["IMG_WIDTH"] = 192
CONFIG_DICT["IMG_CHANNELS"] = 3
CONFIG_DICT["INVERSE_IMG_COLOR"] = False
CONFIG_DICT["CROP_BOTTOM_LEAD"] = False
CONFIG_DICT["EXTRACT_IMG_PATCHES"] = False
CONFIG_DICT["IMAGENET_NORM"] = True  # only work for "torch" and "caffe" mode
CONFIG_DICT["UPSAMPLE_SCALES"] = None
CONFIG_DICT["BATCH_SIZE"] = 32
CONFIG_DICT["NUM_WORKERS"] = 1
CONFIG_DICT["REMOVE_MINORITY_LABELS"] = False
CONFIG_DICT["KEEP_NAN_FEA"] = False
CONFIG_DICT["FEA_NAN_VALUE"] = "mean" # or other value
CONFIG_DICT["EVAL_SPLIT_SIZE"] = 0.2 # the split size for eval dataset
# "white"  or the path of the background image (only work for 'train_gen_img.py')
CONFIG_DICT["BACKGROUND_TYPE"] = "white" 

####### augmentation #######
CONFIG_DICT["AUGMENTATION"] = False
CONFIG_DICT["AUG_PAD_VALUE"] = 1.0
CONFIG_DICT["HEIGHT_SHIFT_RANGE"] = 0.05
CONFIG_DICT["WIDTH_SHIFT_RANGE"] = 0.2
# opt: [-1, 100], -1 means disable
CONFIG_DICT["RANDOM_JPEG_QUALITY"] = -1
# opt: [0, 180], 0 means disable
CONFIG_DICT["RANDOM_ROTATE_ANGLE"] = 0
CONFIG_DICT["RANDOM_CROP_RESIZE_LEAD_WISE_FACTOR"] = -1 # only work for '4x10s' digital
CONFIG_DICT["RANDOM_NOISE_STD"] = -1 # only work for digital now
CONFIG_DICT["RANDOM_SHIFT_LEAD_WISE"] = False # only work for '4x10s' digital
CONFIG_DICT["RANDOM_VERTICAL_SHIFT_LEAD_WISE"] = -1 # only work for 'train_gen_img.py'

####### network #######
# opts: "simple_cnn", "mobile_net", "mobile_net_with_fea"
CONFIG_DICT["NETWORK"] = "mobile_net"
CONFIG_DICT["SIMPLE_WIDER"] = False  # only work for "simple cnn"
CONFIG_DICT["DEPTH_WISE"] = False # group convolution layer
CONFIG_DICT["FROM_SCRATCH"] = False  # only work for pretrained model
CONFIG_DICT["PRETRAINED_WEIGHT"] = "imagenet" # or the path of weight
# opts: None, "concat_early", "concat_later", "concat_only"
CONFIG_DICT["FEATURE_JOINT_STRATEGY"] = None
CONFIG_DICT["SQUEEZE_FEA_FOR_BILINEAR"] = False  # only work for bilinear cnn
CONFIG_DICT["CONCAT_MAXPOOL"] = False
CONFIG_DICT["LOGITS_FOR_LOSS"] = False # only work for `train_v2.py`

####### training #######
CONFIG_DICT["EAGER_EXECUTION"] = False # eager execution mode
# opts: "categorical_crossentropy", "multi_category_focal_loss_v1/2", 
# "categorical_crossentropy_with_pc", "binary_crossentropy", "binary_crossentropy_with_pc"
CONFIG_DICT["LOSS_NAME"] = "categorical_crossentropy"
# Float in [0, 1]. When > 0, label values are smoothed, meaning the confidence on label values are relaxed. 
# e.g. label_smoothing=0.2 means that we will use a value of 0.1 for label 0 and 0.9 for label 1
CONFIG_DICT["LABEL_SMOOTHING"] = 0 # only work for "categorical_crossentropy"
CONFIG_DICT["PAIRWISE_CONFUSION_WEIGHT"] = 1.0 # only work for "categorical_crossentropy_with_pc"
# list for v1, int for v2
CONFIG_DICT["FOCAL_LOSS_ALPHA"] = None
CONFIG_DICT["OPTIMIZER"] = "adam"
CONFIG_DICT["INIT_LR"] = 0.001
CONFIG_DICT["WARM_UP_STEPS"] = False
# opts: "step_decay", "constant"
CONFIG_DICT["LR_STRATEGY"] = "step_decay"
CONFIG_DICT["LR_DECAY_SIZE"] = 15
CONFIG_DICT["TWO_STAGE_FINETUNE"] = True  # only last layer -> all params
CONFIG_DICT["FREEZE_BACKBONE"] = True
CONFIG_DICT["INIT_EPOCH_NUM"] = 20
CONFIG_DICT["FINETUNE_EPOCH_NUM"] = 20
# opts: True, False, [x, y, z, ...]
CONFIG_DICT["CLASS_WEIGHTED"] = False
# False means "zero init", True means "optimal init" ($\log(\frac {n_i} {total})$)
CONFIG_DICT["FINAL_LAYER_INITIAL_BIAS"] = False
CONFIG_DICT["SKIP_EVAL_EPOCH_NUM"] = -1 # only work for `train_v2.py`

####### save #######
CONFIG_DICT["CKPT_PERIOD"] = -1
CONFIG_DICT["CKPT_DIR"] = None
CONFIG_DICT["CKPT_SUFFIX"] = '.h5'
# opts: # "val_categorical_accuracy", "val_acc", "val_binary_accuracy"
CONFIG_DICT["MONITOR_FOR_SAVE"] = "val_categorical_accuracy"  
# opts: True(cache in memory), False(disable cache), "cache/dir"(cache in disk, such as "./tmp")
CONFIG_DICT["CACHE_DIR"] = False  # increase the speed of data pipeline

####### develop net #######
CONFIG_DICT["DEV_ACTIVATION"] = "relu"
CONFIG_DICT["DEV_SELF_ATTN1"] = False
CONFIG_DICT["DEV_SELF_ATTN2"] = False
CONFIG_DICT["DEV_DEPTH_WISE"] = False
CONFIG_DICT["DEV_SELF_ATTN_TYPE"] = "self_attn_with_leads"
CONFIG_DICT["DEV_BLOCK_DIMS"] = [32, 64, 64, 128, 128]
CONFIG_DICT["DEV_FIRST_KERNEL_SIZE"] = 50 # only work for digital models

####### WS-BAN #######
CONFIG_DICT["WS_PART_NUM"] = 32
CONFIG_DICT["WS_DROPOUT_RATE"] = 0.2
CONFIG_DICT["WS_BETA_C"] = 0.05
CONFIG_DICT["WS_LAMBDA_C"] = 1.0
CONFIG_DICT["WS_BACKBONE_NAME"] = "develop_net"
CONFIG_DICT["WS_EXTRA_LAYER_NAME"] = "conv4_block2_out"
CONFIG_DICT["WS_ADD_BAP_NORM"] = True
CONFIG_DICT["WS_ADD_MULTI_HEAD_ATTN"] = False
CONFIG_DICT["WS_MULTI_HEAD_ATTN_NUM"] = 1
CONFIG_DICT["WS_ATTN_HEAD_NUM"] = 8
CONFIG_DICT["WS_MULTI_HEAD_ATTN_ADDING"] = True
CONFIG_DICT["WS_ADD_NORM_DENSE_AFTER_ATTN"] = True
CONFIG_DICT["WS_ADD_DROPOUT_AFTER_ATTN"] = 0.1 # -1 means disable
CONFIG_DICT["WS_ADD_CENTER_LAYER"] = True
# for WS-BAN v2 #
CONFIG_DICT["WS_ADD_DOWN_LAYERS"] = False
CONFIG_DICT["WS_H_NORM_METHOD"] = "softmax" # or "sigmoid", None
CONFIG_DICT["WS_O_NORM_METHOD"] =  "softmax"# or "sigmoid", None
CONFIG_DICT["WS_SP_NORM_METHOD"] = None # or "softmax", "sigmoid"
CONFIG_DICT["WS_CH_NORM_METHOD"] = None # or "softmax", "sigmoid"
CONFIG_DICT["WS_ADD_CHANNEL_ATTN"] = False
CONFIG_DICT["WS_ADD_ATTR_BRANCH"] = False
CONFIG_DICT["WS_GENDER_INDEX"] = 0
CONFIG_DICT["WS_ATTR_WEIGHT"] = 0.3
CONFIG_DICT["WS_ADD_DENSE_LAYER"] = False
CONFIG_DICT["WS_FEAMAP_AS_H"] = False

####### AAAM #######
CONFIG_DICT["AAAM_EMB_DIM"] = 512
CONFIG_DICT["AAAM_DROPOUT"] = 0.5
# the index of gender attribute, -1 means nonexistent
CONFIG_DICT["AAAM_GENDER_INDEX"] = 0
CONFIG_DICT["AAAM_FUSE_REGION_METHOD"] = "max" # or "mean"
CONFIG_DICT["AAAM_CLS_WEIGHT"] = 0.5
CONFIG_DICT["AAAM_ATTR_WEIGHT"] = 0.5
CONFIG_DICT["AAAM_BACKBONE_NAME"] = "develop_net"
CONFIG_DICT["AAAM_FINAL_FEA_TYPE"] = "attr_region"
CONFIG_DICT["AAAM_EXTRA_LAYER_NAME"] = None # only work for "region_attn_net"


####### CNN-RNN with Attn #######
CONFIG_DICT["CRA_EXTRA_LAYER_NAME"] = None
CONFIG_DICT["CRA_RNN_TYPE"] = "gru" # or "lstm"
CONFIG_DICT["CRA_ATTN_TYPE"] = "Bahdanau" # or None or "center_attn" or "center_attn_v2"
CONFIG_DICT["CRA_ADD_LAYER_NORM_AFTER_RNN"] = False
CONFIG_DICT["CRA_BEAM_SEARCH_SIZE"] = -1 # -1 means disable
CONFIG_DICT["CRA_BEAM_SEARCH_NORM_BY_LEN"] = True
CONFIG_DICT["CRA_REGION_ATTN_FOR_CNN"] = False
CONFIG_DICT["CRA_TRAIN_WITH_PLA"] = False
CONFIG_DICT["CRA_CA_ADD_CENTER_LAYER"] = False
CONFIG_DICT["CRA_CA_FEA_DIM"] = 256
CONFIG_DICT["CRA_CA_USE_SPATIAL_HIDDEN"] = False
CONFIG_DICT["CRA_CA_FUSE_HIDDEN_METHOD"] = "add"
CONFIG_DICT["CRA_CA_GATE_FOR_BAP"] = False

# generate default config
cfg_default = dict2config(CONFIG_DICT)
