# eg. python data_analysis.py --csv ../datasets/tianchi_data/tianchi_data_profile_v1.csv --index 3 --save ../datasets/tianchi_data/da_result
import json
import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib


def count_classes_with_sorted(df, class_names):
    class_count = {}
    for name in class_names:
        class_count[name] = df[name].sum()
    class_count_sorted = sorted(
        class_count.items(), key=lambda x: x[1], reverse=True)
    return class_count_sorted


def count_labels_as_csv(df, class_count_sorted):
    gender_map = {0:"male", 1:"female"}
    gender_count = dict(df["Gender"].value_counts())
    genders = [(gender_map[key], gender_count[key]) for key in gender_map]

    age_dict = dict(df["Age"].describe())
    ages = [("Age's "+key, age_dict[key]) for key in ["mean", "std", "min", "max"]]

    lines = [("总样本数", len(df))] + genders + ages + class_count_sorted

    df_count = pd.DataFrame.from_dict(dict(lines), columns=["count"], orient='index')
    return df_count


def plot_co_occurrence_matrix(df, vis_class_names,
                              title="co-occurrence matrix",
                              save_path=None):
    """ plot co-occurrence matrix """
    # build co-occurrence matrix
    def label_relation_a(a, b):
        return ((a+b) == 2).sum()/a.sum()

    def label_relation_b(a, b):
        return ((a+b) == 2).sum()/b.sum()

    corr_df_a = df[vis_class_names].corr(method=label_relation_a)
    corr_df_b = df[vis_class_names].corr(method=label_relation_b)
    mask = np.zeros(corr_df_a.values.shape)
    for i in range(len(mask)):
        for j in range(i, len(mask)):
            mask[i, j] = 1
    corr_df = corr_df_a*mask + corr_df_b*(1-mask)

    # plot
    # if it cannot plot Chinese, follow https://yq.aliyun.com/articles/400452
    # font file "simhei.ttf" can be found in "./tools/"
    sns.set(font="simhei", font_scale=1)
    plt.figure(figsize=(15, 14))
    sns.heatmap(corr_df.round(decimals=2), cmap=plt.cm.YlOrRd,
                square=True, annot=True, annot_kws={"size": 10})
    plt.title(title)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, pad_inches=0.1, dpi=150)
        plt.close()


if __name__ == '__main__':
    from argparse import ArgumentParser
    import os

    parser = ArgumentParser(
        description="Data analysis for ECG labels")
    parser.add_argument("--csv", type=str,
                        help="the path of the input csv file")
    parser.add_argument("--index", type=int, default=29,
                        help="the start class index of in csv columns (3 for tianchi, 29 for ruijin)")
    parser.add_argument("--num", type=int, default=22,
                        help="the #num top classes for plotting (sorted by occurance)")
    parser.add_argument("--title", type=str, default="co-occurrence matrix",
                        help="the title for the co-occurrence matrix")
    parser.add_argument("--save", type=str, default=None,
                        help="the root for saving co-occurrence matrix and statistic result")

    args = parser.parse_args()
    if args.save is not None:
        if not os.path.exists(args.save):
            os.makedirs(args.save)

    df = pd.read_csv(args.csv)
    class_names = list(df.columns[args.index:])
    class_count_sorted = count_classes_with_sorted(df, class_names)

    vis_class_names = [x[0] for x in class_count_sorted[:args.num]]
    fig_path = os.path.join(args.save, "label_relation.png") if args.save is not None else None
    plot_co_occurrence_matrix(df, vis_class_names, args.title, fig_path)

    df_count = count_labels_as_csv(df, class_count_sorted)
    if args.save is not None:
        count_path = os.path.join(args.save, "statistic_result.csv")
        df_count.to_csv(count_path)
