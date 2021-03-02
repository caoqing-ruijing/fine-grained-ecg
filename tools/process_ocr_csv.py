import re
import json
from functools import partial
import pandas as pd
import numpy as np


def replace_label(row, label_relation, multi=False):
    old = [x.strip() for x in row.split("###")]
    if multi:
        new = []
        for x in old:
            if x in label_relation:
                new.extend(label_relation[x])
    else:
        new = [label_relation[x] for x in old]
    new = list(set(new))
    while '' in new:
        new.remove('')
    if len(new) == 0:
        return np.nan
    return "###".join(new)


def split_raw_labels(label_all):
    label_split = []
    null_list = []  # exist ''
    for i, la in enumerate(label_all):
        if "###" in la:
            tmp = [x.strip() for x in la.split("###")]
            if "" in tmp:
                null_list.append(i)
                tmp.remove("")
            label_split.extend(tmp)
        else:
            label_split.append(la.strip())
    return label_split, null_list


def read_ocr_result(csv_path):
    df_all = pd.read_csv(csv_path, na_values=["-1"])  # -1,
    if 'Unnamed: 0' in df_all.columns:
        df_all.pop('Unnamed: 0')
    df_all = df_all[~df_all['Label'].isna()]

    label_all = list(df_all["Label"])
    label_new, _ = split_raw_labels(label_all)
    return df_all, label_new


def extract_easy_labels(label_new):
    # "电轴"、"无法配合"
    dianzhou = [x for x in set(label_new) if ("电轴" in x)]
    dianzhou_right = [x for x in dianzhou if ("右偏" in x) or ('+' in x)]
    dianzhou_left = [x for x in dianzhou if ("左偏" in x)]
    dianzhou_rest = [x for x in dianzhou if (
        "左偏" not in x) and ("右偏" not in x) and ("+" not in x)]
    peihe = [x for x in set(label_new) if ("配合" in x)]
    label_relation = {"电轴右偏": dianzhou_right, "电轴左偏": dianzhou_left,
                      "电轴S1S2S3": dianzhou_rest, "病人无法配合": peihe}

    label_rest = list(set(label_new)-set(dianzhou)-set(peihe))
    return label_relation, label_rest


def del_meaningless_label(df_all, label_relation, label_rest):
    # replace functions
    def replace_pattern1(x):
        x1 = re.sub(r"[（(]?[,，]?请?结合临床[)）]?[;；]?", "", x).strip()
        x2 = re.sub(
            r"([（(]?[,，]?(此图)?与前图[(相比)(比较)(对照)]+，?[(无明显)(略有)]+(动态)?(变化)?[)）]?，?[;；]?)", "", x1).strip()
        x2 = re.sub(
            r"((此图)?与前图比较)|((此图)?[(较)(与)]前图无明显动态[(变化)|(改变)])", "", x2).strip()
        x2 = re.sub(
            r"([（(]?[,，]?建议.+复查[)）]?[;；]?[,，]?)|(，?建议复查)", "", x2).strip()
        x2 = re.sub(r"[（）？]+", "", x2)
        return x2

    def replace_pattern2(x):
        x1 = re.sub(r"(提示)?：?|([（(]+.+[)）]+[;；]?)", "", x).strip()
        x2 = re.sub(r"(图[0-9])+", "", x1).strip()
        return x2

    pc_dict = {}
    for x in label_rest:
        y = replace_pattern1(x)
        y = replace_pattern2(y)
        if y not in pc_dict:
            pc_dict[y] = [x]
        else:
            pc_dict[y].append(x)
    label_relation.update(pc_dict)

    label_relation_reverse = {}
    for key, item in label_relation.items():
        for it in item:
            if it in label_relation_reverse:
                print(it)
            label_relation_reverse[it] = key
    label_relation_reverse[''] = ''

    df_new = df_all.copy()
    fn = partial(replace_label, label_relation=label_relation_reverse)
    df_new["Label"] = df_new["Label"].map(fn)
    df_new = df_new[~df_new['Label'].isna()]
    df_new = df_new.reset_index(drop=True)

    return df_new


def process_numeric_feas(df):

    def fn(item):
        if isinstance(item, str) and "天" in item:
            return True
        else:
            return False

    fea_names = ['Gender', 'Age', 'HR', 'PR', 'QT', 'QTc', 'QRS', 'QRSE']
    target_inds = list(df[df["Age"].map(fn)].index)
    for i in target_inds:
        if "天" in df.loc[i, "Age"]:
            df.loc[i, "Age"] = 1
    for key in fea_names:
        nan_index_list = df[df[key].isin(["/"])].index
        df.loc[nan_index_list, key] = np.nan
        df[key] = pd.to_numeric(df[key])
        # print(key, len(x[x[key].isin(["/"])]), x[key].min(), x[key].max())
    return df


def read_hierarchy_dict(json_path):
    with open(json_path, "r") as f:
        all_hier_dict = json.load(f)

    hier_dict = dict([(key, list(value.keys())) for key, value in all_hier_dict.items()])
    return all_hier_dict, hier_dict


def delete_already_exist(key, low):
    if ',' in low or '，' in low:
        return key

    while low in key:
        key = key.replace(low, '')
    return key


def build_label_relation(df, all_hier_dict, hier_dict):
    """ build label relation according to hierarchy dictionary """
    label_all = list(df["Label"])
    label_split, null_list = split_raw_labels(label_all)
    assert len(null_list) == 0
    label_unique = list(set(label_split))

    label_relation = {}
    for key in label_unique:
        label_relation[key] = []
        key_ori = key
        tops = reversed(list(all_hier_dict.keys()))
        for top in tops:
            mids = reversed(list(all_hier_dict[top]))
            
            for mid in mids:
                lows =list(all_hier_dict[top][mid])
                for low in lows:
                    if '*' in low:
                        m = re.search(low.replace('*', ".*"), key)
                        if m is not None:
                            label_relation[key_ori].append(mid)
                            key = delete_already_exist(key, low)
                            break
                    else:
                        if top == "心肌缺血性诊断":
                            if low==key:
                                label_relation[key_ori].append(mid)
                                break
                            for symbol in ['，','+']:
                                if symbol in key:
                                    for tmp in key.split(symbol):
                                        tmp = tmp.replace('；', '').replace("可能", '')
                                        if low==tmp:
                                            label_relation[key_ori].append(mid)
                        elif low in key:
                            label_relation[key_ori].append(mid)
                            key = delete_already_exist(key, low)
                            break
            if top not in ["T波", "ST段", "ST-T段"]:
                key = key_ori

    return label_relation


def one_hot_labels(df, class_names):
    label_str = list(df["Label"])
    label_np = np.zeros((len(label_str), len(class_names)), dtype=int)
    for i, la in enumerate(label_str):
        las = la.split("###")
        inds = [class_names.index(x) for x in las]
        label_np[i][inds] = 1

    df.pop("Label")
    for i, name in enumerate(class_names):
        df[name] = label_np[:, i]
    return df


def process_inclusion_labels(df):
    """ for each value, set it's key as 1 """
    inclusions = {'房性心动过速': ['房性心动过速伴不等比传导', "短阵房速"],
    'ST段弓背样抬高': ['ST段弓背向上抬高', 'ST段弓背向下抬高'],
    'ST段压低': ['ST段上斜型压低', 'ST段下斜型压低', 'ST段水平型压低'],
    'ST段改变': ['ST段弓背样抬高', 'ST段压低', 'ST段轻度改变'],
    'ST-T段改变': ['ST-T段轻度改变'],
    'T波改变': ['T波高尖', 'T波倒置', 'T波低平', 'T波双向', 'T波轻度改变'],
    '二度房室传导阻滞': ['二度一型房室传导阻滞', '二度二型房室传导阻滞'],
    '二度窦房传导阻滞': ['二度一型窦房传导阻滞', '二度二型窦房传导阻滞']}

    for key,value in inclusions.items():
        for v in value:
            df.loc[df[df[v]==1].index, key] = 1
    return df


def main(csv_path, json_path, save_path=None):
    df_all, label_new = read_ocr_result(csv_path)
    label_relation, label_rest = extract_easy_labels(label_new)
    df_new = del_meaningless_label(df_all, label_relation, label_rest)
    df_new = process_numeric_feas(df_new)

    all_hier_dict, hier_dict = read_hierarchy_dict(json_path)
    label_relation = build_label_relation(df_new, all_hier_dict, hier_dict)
    
    df_new = df_new.copy()
    fn = partial(replace_label, label_relation=label_relation, multi=True)
    df_new["Label"] = df_new["Label"].map(fn)
    df_new = df_new[~df_new['Label'].isna()]

    class_names = [item for sublist in hier_dict.values() for item in sublist]
    df_new = one_hot_labels(df_new, class_names)
    df_new = process_inclusion_labels(df_new)

    if save_path is not None:
        df_new.to_csv(save_path, index=False)
    return df_new

if __name__ == "__main__":
    from argparse import ArgumentParser
    import multiprocessing
    from functools import partial

    parser = ArgumentParser(
        description="process the result of 'OCR_reader_mp.py'")
    parser.add_argument("--csv", type=str,
                        help="the path of OCR result csv file")
    parser.add_argument("--json", type=str,
                        help="the path of json file which contains label hierarchy")
    parser.add_argument("--save", type=str,
                        help="the path for saving the processed csv file")

    args = parser.parse_args()
    main(args.csv, args.json, args.save)