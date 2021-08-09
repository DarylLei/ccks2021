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

import os,sys
sys.path.append('.')
import os.path as osp
import codecs
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--test_path", type=str, default="/data1/lvzhengwei/git_code/torch_dir/multi_modal/ccks/data/ccks2021/test_b.json")
parser.add_argument(
    "--category_level1_result",
    type=str,
    default='ccks/submission/level1_en.json')
    # default='ccks/output_text_pre2/result/tag_cls/level1/level1-Iter-11200-acc-0.7493.pth.json')
parser.add_argument(
    "--category_level2_result",
    type=str,
    default='ccks/submission/level2_en.json')
    # default='ccks/output_text_pre2/result/tag_cls/level2/level2-Iter-14400-acc-0.5753.pth.json')
parser.add_argument(
    "--tag_result",
    type=str,
    default='ccks/submission/sem_en.json')
    # default='ccks/output_text_pre2/result/sem_tag/Iter-17600-acc-0.3928.pth.json')

def get_cls_series():
    fir_2_sec = {}
    sec_2_fir = {}
    with codecs.open('ccks/data/ccks2021/topic.json','r','utf-8') as fr:
        for line in fr:
            tmp_data = json.loads(line.strip())
            fir_level = tmp_data["@value"]
            sec_level = [ _item["@value"] for _item in tmp_data["belong"]]
            fir_2_sec[fir_level] = sec_level
            for _item in sec_level:
                if _item == '其他':
                    sec_2_fir[fir_level + '_' + _item] = fir_level
                else:
                    sec_2_fir[_item] = fir_level
    return fir_2_sec,sec_2_fir


if __name__ == "__main__":
    # fir_2_sec,sec_2_fir = get_cls_series()
    args = parser.parse_args()
    print(args)

    with codecs.open(args.test_path, "r", encoding="utf-8") as inf:
        print("Loading {}...".format(args.test_path))
        lines = inf.readlines()
        nids = [json.loads(line)["@id"] for line in lines]

    # load the prediction results of 'paddle-video-classify-tag' model on test-set
    with codecs.open(
            args.category_level1_result, "r", encoding="utf-8") as inf:
        pred_level1 = json.load(inf)
    with codecs.open(
            args.category_level2_result, "r", encoding="utf-8") as inf:
        pred_level2 = json.load(inf)
    # load the prediction results of 'paddle-video-semantic-tag' model on test-set
    with codecs.open(args.tag_result, "r", encoding="utf-8") as inf:
        pred_tags = json.load(inf)

    # merge results and generate an entry for each nid.
    submission_lines = []
    for nid in nids:
        level1_category = pred_level1[nid]["class_name"] \
                          if nid in pred_level1 else ""
        level2_category = pred_level2[nid]["class_name"] \
                          if nid in pred_level2 else ""
        tags = pred_tags[nid] if nid in pred_tags else []
        result = {
            "@id": nid,
            "category": [
                {
                    "@meta": {
                        "type": "level1"
                    },
                    "@value": level1_category
                },
                {
                    "@meta": {
                        "type": "level2"
                    },
                    "@value": level2_category
                },
            ],
            "tag": [{
                "@value": tag
            } for tag in tags],
        }
        submission_lines.append(json.dumps(result, ensure_ascii=False) + "\n")

    with codecs.open("ccks/submission/result.txt", "w", encoding="utf-8") as ouf:
        ouf.writelines(submission_lines)
    print("Saved result.txt")
