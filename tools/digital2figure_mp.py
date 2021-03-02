# -*- coding: utf-8 -*-
# eg. python digital2figure_mp --root ../datasets/tianchi_data/train/ --output ../datasets/tianchi_data/hf_fig_noise/ --crop --cpu 8

import os
import json
import glob
import pandas as pd
import numpy as np
from scipy.signal import butter, lfilter
from PIL import Image, ImageDraw, ImageFont


def make_pts(xs, curve):
    pts = np.concatenate([xs[:,None], curve[:,None]], axis=1)
    return list(map(tuple, pts))


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def digital2figure_single_txt(file_path, save_dir, bg, crop=False):
    # read file
    channel_list = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    # 这个会直接默认读取到这个Excel的第一个表单
    df = pd.read_csv(file_path, sep=' ', names=channel_list, skiprows=[0])

    # build leads dict
    tmp_dict = {}
    for each_channel in channel_list:
        tmp_dict[each_channel] = list()
    for index, row in df.iterrows():
        for each_channel in channel_list:
            tmp_dict[each_channel].append(row[each_channel])

    # calculate the rest leads
    tmp_dict['III'] = np.asarray(tmp_dict['II']) - np.asarray(tmp_dict['I'])
    tmp_dict['aVR'] = -(np.asarray(tmp_dict['I']) + np.asarray(tmp_dict['II']))/2
    tmp_dict['aVL'] = np.asarray(tmp_dict['I']) - np.asarray(tmp_dict['II'])/2
    tmp_dict['aVF'] = np.asarray(tmp_dict['II']) - np.asarray(tmp_dict['I'])/2

    # butter bandpass filter
    fs = 500.0
    lowcut = 2.0
    highcut = 50.0
    for c in (channel_list+['III','aVR','aVL','aVF']):
        tmp_dict[c] = butter_bandpass_filter(tmp_dict[c], lowcut, 
                                             highcut, fs, order=6)

    # curve_1
    curve_1 = np.zeros(5000)
    curve_1[0:1250] = np.asarray(tmp_dict['I'])[0:1250]
    curve_1[1250:2500] = np.asarray(tmp_dict['aVR'])[1250:2500]
    curve_1[2500:3750] = np.asarray(tmp_dict['V1'])[2500:3750]
    curve_1[3750:5000] = np.asarray(tmp_dict['V4'])[3750:5000]
    curve_1 = curve_1 + 500
    # curve_2
    curve_2 = np.zeros(5000)
    curve_2[0:1250] = np.asarray(tmp_dict['II'])[0:1250]
    curve_2[1250:2500] = np.asarray(tmp_dict['aVL'])[1250:2500]
    curve_2[2500:3750] = np.asarray(tmp_dict['V2'])[2500:3750]
    curve_2[3750:5000] = np.asarray(tmp_dict['V5'])[3750:5000]
    # curve_3
    curve_3 = np.zeros(5000)
    curve_3[0:1250] = np.asarray(tmp_dict['III'])[0:1250]
    curve_3[1250:2500] = np.asarray(tmp_dict['aVF'])[1250:2500]
    curve_3[2500:3750] = np.asarray(tmp_dict['V3'])[2500:3750]
    curve_3[3750:5000] = np.asarray(tmp_dict['V6'])[3750:5000]
    curve_3 = curve_3 - 500
    # curve_4
    curve_4 = tmp_dict['II'] - 1000

    # plot
    img = bg.copy()
    draw = ImageDraw.Draw(img)
    xs = np.linspace(180, 2806-250, 5000)
    draw.line(make_pts(xs, 1170-curve_1), width=2)
    draw.line(make_pts(xs, 1000-curve_2), width=2)
    draw.line(make_pts(xs, 830-curve_3), width=2)
    draw.line(make_pts(xs, 660-curve_4), width=2)

    if crop:
        img = img.crop([107, 477, 2625, 1816])
    # save
    img_name = os.path.basename(file_path).split('.')[0]+'.png'
    img.save(os.path.join(save_dir, img_name))


def digital2figure_single_json(file_path, save_dir, bg, crop=False):
    """ for ruijin refine json"""
    with open(file_path, 'r') as f:
        result = json.load(f)

    leads_inline = [
        ['I','aVR','V1','V4'],
        ['I I','aVL','V2','V5'],
        ['III','aVF','V3','V6'],
        ['II']]
    leads_inline_gap = [1415+695*i for i in range(4)] 
    leads_inline_txt_gap = [1215+695*i for i in range(4)]

    s=2.1
    img = bg.resize((int(2806*s), int(1984*s)), Image.BILINEAR)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("./simhei.ttf", 60)
    mult_r = 7
    for i in range(len(leads_inline)):
        lead_inline = leads_inline[i]
        current_pt = 370

        for j in range(len(lead_inline)):
            key = lead_inline[j]
            raw_signal = result[key]['value']

            raw_signal = list(np.array(raw_signal)/-1)
            raw_signal_y = list(np.array(raw_signal)*mult_r+leads_inline_gap[i])
            raw_signal_x = [k+current_pt+1250*j for k in range(len(raw_signal_y))]
            real_pts = list(zip(raw_signal_x, raw_signal_y))

            txt_pt = (current_pt+1250*j,leads_inline_txt_gap[i])
            draw.text(txt_pt, key, fill='black',font=font)
            draw.line(real_pts,fill='black', width=5)
            if key not in ['I', 'I I', 'III', 'II']:
                line_pts = [
                    (current_pt+1250*j,leads_inline_txt_gap[i]-20),
                    (current_pt+1250*j,leads_inline_txt_gap[i]+400)]
                draw.line(line_pts,fill='black', width=7)
    # resize
    w, h = img.size
    img = img.resize((w//5, h//5), resample=Image.ANTIALIAS)

    if crop:
        w, h = img.size
        crop_box = [107/2806*w, 477/1984*h, 2625/2806*w, 1816/1984*h]
        crop_box = list(map(int, crop_box))
        img = img.crop(crop_box)
    # save
    img_name = os.path.basename(file_path).split('.')[0]+'.png'
    img.save(os.path.join(save_dir, img_name))

if __name__ == '__main__':
    from argparse import ArgumentParser
    import multiprocessing
    from tqdm import tqdm

    parser = ArgumentParser(
        description="plot ECG digital as figure in mulitprocess")
    parser.add_argument("--root", type=str,
                        help="the root of input files (.txt or .json)")
    parser.add_argument("--output", type=str,
                        help="the directory for saving output figures")
    parser.add_argument("--denoise", action="store_true",
                        help="use white background for plotting")
    parser.add_argument("--crop", action="store_true",
                        help="crop the figure and only keep the grid")
    parser.add_argument("--suffix", type=str, default="txt",
                        help="tianchi(.txt) or ruijin(.json)")
    parser.add_argument("--cpu", type=int, default=4,
                        help="the cores of cpu")
    parser.add_argument("--background", type=str, default="./ruijing_background.png",
                        help="the path of the background")

    args = parser.parse_args()

    # input folder for digital files
    root = args.root 
    save_dir = args.output
    crop = args.crop
    # background
    if args.denoise:
        bg = Image.fromarray(np.ones((1984,2806), dtype=np.uint8)*255)
    else:
        bg = Image.open(args.background).copy()
    if args.suffix=="txt":
        fn = digital2figure_single_txt
    elif args.suffix=="json":
        fn = digital2figure_single_json
    else:
        raise NotImplementedError("%s is not implemented now"%args.suffix)

    def process_fn(file_path):
        global save_dir, bg, crop, fn
        try:
            fn(file_path, save_dir, bg, crop)
        except Exception as e:
            print(e)

    file_path_list = glob.glob(os.path.join(root,"*%.s"%args.suffix))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    pool = multiprocessing.Pool(processes=args.cpu)
    with tqdm(total=len(file_path_list)) as progress_bar:
        for _ in pool.imap_unordered(process_fn, file_path_list):
            progress_bar.update(1)
    print('Done')
