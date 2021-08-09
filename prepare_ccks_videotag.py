# Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import os.path as osp
from logger import logger
import codecs
import json
import argparse
import random
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument(
    "--trainval_path", type=str, default="/data1/lvzhengwei/git_code/torch_dir/multi_modal/ccks/data/ccks2021/train.json")
parser.add_argument(
    "--test_path", type=str, default="/data1/lvzhengwei/git_code/torch_dir/multi_modal/ccks/data/ccks2021/test_b.json")
parser.add_argument(
    "--trainval_tsn_feature_dir",
    type=str,
    default="/data1/lvzhengwei/git_code/torch_dir/ccks_data/Data/CCKS_dataset/tsn_features_train")
# parser.add_argument(
#     "--test_tsn_feature_dir",
#     type=str,
#     default="CCKS_dataset/tsn_features_test_a")

parser.add_argument(
    "--trainval_text_dir",
    type=str,
    default="data/text_data/trainval")
parser.add_argument(
    "--test_text_dir",
    type=str,
    default="data/text_data/test_b")


def create_splits_indice(n_samples, SPLITS):
    assert sum([v for k, v in SPLITS]) == 1.0
    indices = list(range(n_samples))
    random.shuffle(indices)
    split2indice = {}
    r_offset = 0
    for idx, (split, ratio) in enumerate(SPLITS):
        l_offset = r_offset
        if idx == len(SPLITS) - 1:
            r_offset = n_samples
        else:
            r_offset = int(n_samples * ratio) + l_offset
        split2indice[split] = indices[l_offset:r_offset]
    return split2indice


def prepare_split(data, split_name, test_only=False, gather_labels=False):
    '''
      1. Prepare ALL (unique) labels for classification from trainval-set.
      2. For each split, generate sample list for level1 & level2 classification.
    '''
    sample_nids = [sample["@id"] for sample in data]
    level1_labels = []
    level2_labels = []
    if not test_only:
        for sample in data:
            category = {
                each["@meta"]["type"]: each["@value"]
                for each in sample["category"]
            }
            level1_labels.append(category["level1"])
            level2_labels.append(category["level2"])

    def create_sample_list(sample_labels, level_name):
        save_label_file = "ccks/data/npy_pre/{}_label.txt".format(level_name)
        if gather_labels:
            # For trainval set:
            # Gather candidate labels and dump to {level1,level2}_label.txt
            labels = sorted([str(label) for label in list(set(sample_labels))])
            with codecs.open(save_label_file, "w", encoding="utf-8") as ouf:
                ouf.writelines([label + "\n" for label in labels])
                print("Saved " + save_label_file)
        else:
            # For test set: load existing labels.
            with codecs.open(save_label_file, "r", encoding="utf-8") as inf:
                labels = [line.strip() for line in inf.readlines()]
        label2idx = {label: idx for idx, label in enumerate(labels)}
        sample_lines = []
        # Generate sample list: one sample per line (feature_path -> label)
        for i in range(len(sample_nids)):
            label_indice = label2idx[str(sample_labels[i])] if not test_only \
                           else -1
            if split_name in ["train", "val", "trainval"]:
                tsn_feature_dir = args.trainval_tsn_feature_dir
            elif split_name in ["test"]:
                tsn_feature_dir = args.test_tsn_feature_dir
            feature_path = osp.join(tsn_feature_dir,
                                    "{}.npy".format(sample_nids[i]))
            if osp.exists(feature_path):
                line = "{} {}\n".format(feature_path, str(label_indice))
                sample_lines.append(line)
        save_split_file = "ccks/data/npy_pre/{}_{}.list".format(level_name, split_name)
        with codecs.open(save_split_file, "w", encoding="utf-8") as ouf:
            ouf.writelines(sample_lines)
            print("Saved {}, size={}".format(save_split_file,
                                             len(sample_lines)))

    create_sample_list(level1_labels, "level1")
    create_sample_list(level2_labels, "level2")


def prepare_text_data(data, split_name, dir_save, test_only=False, gather_labels=False):
    '''
      1. Prepare ALL (unique) labels for classification from trainval-set.
      2. For each split, generate sample list for level1 & level2 classification.
    '''
    sample_nids = [sample["@id"] for sample in data]
    level1_labels = []
    level2_labels = []
    if not test_only:
        for sample in data:
            category = {
                each["@meta"]["type"]: each["@value"]
                for each in sample["category"]
            }
            level1_labels.append(category["level1"])
            if category["level2"] == '其他':
                level2_labels.append(category["level1"]+'_'+category["level2"])
            else:
                level2_labels.append(category["level2"])

    def create_sample_list(sample_labels, level_name):
        save_label_file = dir_save + "/{}_label.txt".format(level_name)
        if gather_labels:
            # For trainval set:
            # Gather candidate labels and dump to {level1,level2}_label.txt
            # labels = sorted([str(label) for label in list(set(sample_labels))])
            if level_name == 'level1':
                labels = sorted([str(label) for label in list(set(sample_labels))])
                special_label = []
                labels_static = Counter(sample_labels)
                json.dump(dict(labels_static),codecs.open(dir_save + "/{}_labels_static.json".format(level_name),'w'),
                                ensure_ascii=False,indent=2)  
                for key,value in dict(labels_static).items():
                    if value < 100:
                        special_label.append(key)  
                print(special_label)       
            else:
                labels = sorted([str(label) for label in list(set(sample_labels))])
                labels_static = Counter(sample_labels)
                json.dump(dict(labels_static),codecs.open(dir_save + "/{}_labels_static.json".format(level_name),'w'),
                             ensure_ascii=False,indent=2)
            with codecs.open(save_label_file, "w", encoding="utf-8") as ouf:
                ouf.writelines([label + "\n" for label in labels])
                print("Saved " + save_label_file)
        else:
            # For test set: load existing labels.
            with codecs.open(save_label_file, "r", encoding="utf-8") as inf:
                labels = [line.strip() for line in inf.readlines()]
        label2idx = {label: idx for idx, label in enumerate(labels)}
        sample_lines = []
        # Generate sample list: one sample per line (feature_path -> label)
        for i in range(len(sample_nids)):
            label_indice = label2idx[str(sample_labels[i])] if not test_only \
                           else -1
            if split_name in ["train", "val", "trainval"]:
                tsn_feature_dir = args.trainval_text_dir
            elif split_name in ["test"]:
                tsn_feature_dir = args.test_text_dir
            feature_path = osp.join(tsn_feature_dir,
                                    "{}.txt".format(sample_nids[i]))
            if osp.exists(feature_path):
                line = "{} {}\n".format(feature_path, str(label_indice))
                sample_lines.append(line)
        save_split_file = dir_save + "/{}_{}.list".format(level_name, split_name)
        with codecs.open(save_split_file, "w", encoding="utf-8") as ouf:
            ouf.writelines(sample_lines)
            print("Saved {}, size={}".format(save_split_file,
                                             len(sample_lines)))

    create_sample_list(level1_labels, "level1")
    create_sample_list(level2_labels, "level2")




if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    random.seed(66662)

    # load data for train & validation (have labels).
    with codecs.open(args.trainval_path, "r", encoding="utf-8") as inf:
        print("Loading {}...".format(args.trainval_path))
        lines = inf.readlines()
        trainval_data = [json.loads(line) for line in lines]

    # load data for test (no labels).
    with codecs.open(args.test_path, "r", encoding="utf-8") as inf:
        print("Loading {}...".format(args.test_path))
        lines = inf.readlines()
        test_data = [json.loads(line) for line in lines]

    # split the trainval data into train-set(80%) and validation-set(20%).
    split2indice = create_splits_indice(
        len(trainval_data), [
            ("train", 4.0 / 5.0),
            ("val", 1.0 / 5.0),
        ])
    train_data = [trainval_data[idx] for idx in split2indice["train"]]
    val_data = [trainval_data[idx] for idx in split2indice["val"]]

    # prepare_split(trainval_data, "trainval", gather_labels=True)
    # prepare_split(train_data, "train")
    # prepare_split(val_data, "val")
    # prepare_split(test_data, "test", test_only=True)

    dir_save = 'data/class_tag/text_pre5'
    if not os.path.exists(dir_save):
      os.makedirs(dir_save)
      logger.info('create logger path:{}'.format(dir_save))
    prepare_text_data(trainval_data, "trainval", dir_save, gather_labels=True)
    prepare_text_data(train_data, "train", dir_save)
    prepare_text_data(val_data, "val", dir_save)
    prepare_text_data(test_data, "test", dir_save, test_only=True)

