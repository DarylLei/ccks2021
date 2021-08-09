import sys
sys.path.append('.')
import codecs,json,glob
import numpy as np
from collections import Counter
from metrics import get_add_sem_tag


def avg_ensem_level1(level):
  data_all = {}
  if level == 'level1':
    files2 = glob.glob('ccks/output_text_pre2/result/tag_cls/level1/*')
    files3 = glob.glob('ccks/output_text_pre3/result/tag_cls/level1/*')
    files4 = glob.glob('ccks/output_text_pre4/result/tag_cls/level1/*')
    files5 = glob.glob('ccks/output_text_pre5/result/tag_cls/level1/*')
    path_save = 'ccks/submission/level1_en.json'
  elif level == 'level2':
    files2 = glob.glob('ccks/output_text_pre2/result/tag_cls/level2/*')
    files3 = glob.glob('ccks/output_text_pre3/result/tag_cls/level2/*')
    files4 = glob.glob('ccks/output_text_pre4/result/tag_cls/level2/*')
    files5 = glob.glob('ccks/output_text_pre5/result/tag_cls/level2/*')
    path_save = 'ccks/submission/level2_en.json'
  files = files2 + files3 + files4 + files5
  print(len(files))
  for item in files:
    with codecs.open(item,'r','utf-8') as fr:
      print(item)
      data_json = json.load(fr)
      for key,value in data_json.items():
        class_name = value['class_name']
        if key in data_all:
          data_all[key].append(class_name)
        else:
          data_all[key] = [class_name]

  output_result = {}
  for key,value in data_all.items():
    value_counter = Counter(value)
    value_s = value_counter.most_common(1) 
    output_result[key] = {"class_name":value_s[0][0]}
    # 保存结果
  with codecs.open(path_save,'w',"utf-8") as f:
      json.dump(output_result, f, ensure_ascii=False, indent = 4)
  print('predict finished !')

def avg_ensem_sem():
  data_all = {}
  files1 = glob.glob('ccks/output_add_asr1/result/sem_tag/*')
  files2 = glob.glob('ccks/output_add_asr2/result/sem_tag/*')
  files3 = glob.glob('ccks/output_add_asr3/result/sem_tag/*')
  files4 = glob.glob('ccks/output_add_asr4/result/sem_tag/*')
  files4 = glob.glob('ccks/output_add_asr5/result/sem_tag/*')
  files = files2 + files3 + files4 + files1
  print(len(files))
  for item in files:
    print(item)
    with codecs.open(item,'r','utf-8') as fr:
      data_json = json.load(fr)
      for key,value in data_json.items():
        value = [sub_v.strip() for sub_v in value]
        if key in data_all:
          data_all[key].extend(value)
        else:
          data_all[key] = value

  output_result = {}
  for key,value in data_all.items():
    value_new = []
    value_counter = Counter(value)
    value_s = value_counter.most_common()
    for _sub_key,_num in value_s:
      if _num >1:
        value_new.append(_sub_key)
    output_result[key] = value_new
  # 保存结果
  path_save = 'ccks/submission/sem_en.json'
  with codecs.open(path_save,'w',"utf-8") as f:
      json.dump(output_result, f, ensure_ascii=False, indent = 4)
  print('predict finished !')

def eval_ensem():
  data_all = {}
  files2 = glob.glob('ccks/ensem_test_text_pre2/result/sem_tag/*')
  files = files2 
  for item in files:
    print(item)
    with codecs.open(item,'r','utf-8') as fr:
      data_json = json.load(fr)
      for key,value in data_json.items():
        value = [sub_v.strip() for sub_v in value]
        if key in data_all:
          data_all[key].extend(value)
        else:
          data_all[key] = value

  output_result = {}
  for key,value in data_all.items():
    value_new = []
    value_counter = Counter(value)
    value_s = value_counter.most_common()
    for _sub_key,_num in value_s:
      if _num >1:
        value_new.append(_sub_key)
    output_result[key] = value_new

  y_true = []
  y_pre = []
  tmp_version = 'add_asr2'
  with open("ccks/data/sem_tag_pre/{}/val_nids.txt".format(tmp_version), "r") as inf:
      id_lines = inf.readlines()
      for item in id_lines:
          _id = item.strip().split('\t')[0]
          _tags = json.loads(item.strip().split('\t')[1])
          _tags = [ _id+"_"+_item["@value"]for _item in _tags]
          y_true.extend(_tags)

  for key,value in output_result.items():
      _tags = [ key+"_"+_item for _item in value]
      y_pre.extend(_tags)

  true_entities = set(y_true)
  pred_entities = set(y_pre)
  # tmp_c = true_entities & pred_entities
  nb_correct = len(true_entities & pred_entities)
  nb_pred = len(pred_entities)
  nb_true = len(true_entities)

  p = nb_correct / nb_pred if nb_pred > 0 else 0
  r = nb_correct / nb_true if nb_true > 0 else 0
  score = 2 * p * r / (p + r) if p + r > 0 else 0
  print('p: {:.4}  r: {:.4} f1: {:.4} '.format(p,r,score))


if __name__=='__main__':
  ## 集成一级分类标签结果
  avg_ensem_level1('level1')
  ## 集成二级分类标签结果
  avg_ensem_level1('level2')
  ## 集成语义标签结果
  avg_ensem_sem()
  ## 评估语义标签集成效果
  # eval_ensem()

