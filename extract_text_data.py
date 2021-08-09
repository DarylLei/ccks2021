from itertools import count
import os,sys
sys.path.append('.')
import re
import codecs
import json
import argparse
import random
from collections import Counter
from logger import logger


def get_text_data(path_r,dir_save):

    if not os.path.exists(dir_save):
      os.makedirs(dir_save)
      logger.info('create logger path:{}'.format(dir_save))

    ### 检验测试数据中是否有重复id 的数据。
    # with codecs.open(args.test_path, "r", encoding="utf-8") as inf:
    #     print("Loading {}...".format(args.test_path))
    #     lines = inf.readlines()
    #     nids = [json.loads(line)["@id"] for line in lines]
    #     nids_set = Counter(nids)
    #     repeat_id = nids_set.most_common(30)
    #     repeat_nid = [item[0] if item[1] >1 else " "  for item in repeat_id]
    #     print(repeat_nid)

    ### 提取文本数据到单个文件中
    with codecs.open(path_r, "r", encoding="utf-8") as inf:
        logger.info("Loading {}...".format(path_r))
        lines = inf.readlines()
        for i, line in enumerate(lines):
          if i%1000 == 0 and i>0:
            logger.info(i)
          tmp_data = json.loads(line)
          title = tmp_data['title']
          title = re.sub(' ','',title)
          data_id = tmp_data['@id']
          asr_data = tmp_data['perception']['asr']
          asr_text = []
          for item in asr_data:
            for sub_item in item:
              if sub_item not in asr_text:
                asr_text.append(sub_item) 
          asr_text = ','.join(asr_text)
          ocr_text = []
          ocr_data = tmp_data['perception']['ocr']
          for item in ocr_data:
            for sub_item in item:
              if sub_item['word'] not in ocr_text:
                ocr_text.append(sub_item['word'])
          ocr_text = ','.join(ocr_text)
          with codecs.open(dir_save + data_id + '.txt','w','utf-8') as fw:
            fw.write(json.dumps({'title':title,'asr':asr_text,'ocr':ocr_text}))
        print(i)


if __name__ == "__main__":

    random.seed(6666)
    # load data for train & validation (have labels).
    get_text_data('/data1/lvzhengwei/git_code/torch_dir/multi_modal/ccks/data/ccks2021/train.json','data/text_data/trainval/')
    get_text_data('/data1/lvzhengwei/git_code/torch_dir/multi_modal/ccks/data/ccks2021/test_b.json','data/text_data/test_b/')
