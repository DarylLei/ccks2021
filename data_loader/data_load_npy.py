import glob,codecs
import numpy as np
from logger import logger
from torch.utils.data import Dataset, DataLoader
import torch
import json
from transformers import BertTokenizer
import random

def get_label_name_dict(path):
  with codecs.open(path,'r','utf-8') as inf:
      labels = [line.strip() for line in inf.readlines()]
      label_2_id = {label: idx for idx, label in enumerate(labels)}
      id_2_label = {idx: label for idx, label in enumerate(labels)}
  return label_2_id,id_2_label


def get_data(data_path,npy_path_dir,debug_flag=False):
  npy_data = []
  text_data = []
  data_labels= []
  with codecs.open(data_path,'r','utf-8') as fr:
    for line in fr:
      text_path = line.strip().split()[0]
      label = int(line.strip().split()[1]) 
      label = torch.tensor(label, dtype=torch.long)
      data_labels.append(label)
      npy_path = npy_path_dir + text_path.strip().split('/')[-1][:-4] + '.npy'
      npy_data.append(npy_path)
      text_data.append(text_path)

  logger.info('load npy data ...')
  npy_feature = get_npy_feature(npy_data,debug_flag)
  logger.info('load text data ...')
  text_feature = get_text_feature(text_data,debug_flag)
  if debug_flag:
    return npy_feature, text_feature, data_labels[:501]
  else:
    return npy_feature,text_feature,data_labels

def get_text_feature(text_data,debug_flag,MAX_LEN=300):
    word_tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
    features = []
    for i, item in enumerate(text_data):
      if debug_flag and i>500:
        break
      if i%1000 == 0 and i>0:
        logger.info(i)
      with codecs.open(item,'r','utf-8') as fr_2:
        line = fr_2.read()
        data_json = json.loads(line)
        ocr = data_json['ocr']
        asr = data_json['asr']
        title = data_json['title']
        comment_text = title + '。' + asr[:100] + ocr[-100:]
        comment_text = " ".join(comment_text.split())

        inputs = word_tokenizer(
            comment_text,
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True
        )
        ids = torch.tensor(inputs['input_ids'], dtype=torch.long)
        mask = torch.tensor(inputs['attention_mask'], dtype=torch.long) 
        token_type_ids = torch.tensor(inputs["token_type_ids"], dtype=torch.long) 
        features.append({'ids':ids,'mask':mask,'token_type_ids':token_type_ids})
    return features

def get_npy_feature(npy_feature_path,debug_flag):
  npy_feature = []
  for i, item in enumerate(npy_feature_path):
    if i>500 and debug_flag:
      break
    if i%1000 == 0 and i>0:
      logger.info(i)
    try:
      frames = np.load(item)
      frames_select = frames[0,:]
    except:
      frames_select = np.array([0.]*2048)
      logger.info('data error :{}'.format(item))
    features = torch.tensor(frames_select, dtype=torch.float)
    npy_feature.append(features)
  return npy_feature

def build_dataloader(config,debug_flag,level,data_version):
  TRAIN_BATCH_SIZE = config.train_batch_size
  VALID_BATCH_SIZE = config.valid_batch_size
  if level == 'level1':
    train_data_path = 'data/class_tag/{}/level1_train.list'.format(data_version)
    valid_data_path = 'data/class_tag/{}/level1_val.list'.format(data_version)
  elif level == 'level2':
    train_data_path = 'data/class_tag/{}/level2_train.list'.format(data_version)
    valid_data_path = 'data/class_tag/{}/level2_val.list'.format(data_version)
  logger.info(train_data_path)
  # train_data = get_data_add_version(data_version,level,train_data_path,debug_flag=debug_flag)
  train_data = get_data(train_data_path,config.npy_path_dir_train,debug_flag=debug_flag)
  valid_data = get_data(valid_data_path,config.npy_path_dir_train,debug_flag=debug_flag)
  
  


  logger.info("TRAIN Dataset: {}".format(len(train_data[0])))
  logger.info("VALID Dataset: {}".format(len(valid_data[0])))

  train_set = CustomDataset(train_data)
  valid_set = CustomDataset(valid_data)

  train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 2
                }
  valid_params = {'batch_size': VALID_BATCH_SIZE,
                  'shuffle': False,
                  'num_workers': 2
                  }

  train_loader = DataLoader(train_set, **train_params)
  valid_loader = DataLoader(valid_set, **valid_params)

  return train_loader,valid_loader

class CustomDataset(Dataset):
  def __init__(self, data):
      self.npy_features, self.text_features,self.data_labels = data

  def __len__(self):
      return len(self.data_labels)

  def __getitem__(self, index):
      text_feature = self.text_features[index]
      labels = self.data_labels[index]
      npy_feature = self.npy_features[index]
      return {
          'text_feature': text_feature,
          'npy_feature': npy_feature,
          'labels':labels
      }

class CustomValDataset(Dataset):
  def __init__(self, data):
      self.npy_features, self.text_features,self.data_labels,self.sample_ids = data

  def __len__(self):
      return len(self.data_labels)

  def __getitem__(self, index):
      text_feature = self.text_features[index]
      labels = self.data_labels[index]
      npy_feature = self.npy_features[index]
      _id = self.sample_ids[index]
      return {
          'text_feature': text_feature,
          'npy_feature': npy_feature,
          'video_id':_id,
          'labels':labels
      }




def get_predict_data(config,debug_flag,level,data_version):
  Test_BATCH_SIZE = 16
  if level == 'level1':
    test_data_path = 'data/class_tag/{}/level1_test.list'.format(data_version)
  elif level == 'level2':
    test_data_path = 'data/class_tag/{}/level2_test.list'.format(data_version)
  logger.info(test_data_path)
  video_ids = []
  with codecs.open(test_data_path,'r','utf-8') as fr:
    for line in fr:
      text_path = line.strip().split()[0]
      sample_id  = text_path.split('/')[-1][:-4]
      video_ids.append(sample_id)
  npy_path_dir = config.npy_path_dir_test  
  test_data = get_data(test_data_path,npy_path_dir,debug_flag)
  if debug_flag:
    video_ids = video_ids[:1001]

  logger.info("Test Dataset: {}".format(len(test_data[0])))

  test_set = CustomValDataset([test_data[0],test_data[1],test_data[2],video_ids])

  test_params = {'batch_size': Test_BATCH_SIZE,
                'shuffle': False,
                'num_workers': 2
                }

  test_loader = DataLoader(test_set, **test_params)

  return test_loader


def get_val_data(config,debug_flag,level,data_version):
  if level == 'level1':
    valid_data_path = 'data/class_tag/{}/level1_val.list'.format(data_version)
  elif level == 'level2':
    valid_data_path = 'data/class_tag/{}/level2_val.list'.format(data_version)

  video_ids = []
  with codecs.open(valid_data_path,'r','utf-8') as fr:
    for line in fr:
      text_path = line.strip().split()[0]
      sample_id  = text_path.split('/')[-1][:-4]
      video_ids.append(sample_id)    
  valid_data = get_data(valid_data_path,config.npy_path_dir_train,debug_flag=debug_flag)
  if debug_flag:
    video_ids = video_ids[:1001]
  logger.info("val Dataset: {}".format(len(video_ids)))

  val_set = CustomValDataset([valid_data[0],valid_data[1],valid_data[2],video_ids])

  val_params = {'batch_size': 8,
                  'shuffle': False,
                  'num_workers': 2
                  }

  val_loader = DataLoader(val_set, **val_params)

  return val_loader



## 数据增强版本
def get_text_feature_add(data_labels,time_result,text_data,debug_flag,MAX_LEN=400):
    word_tokenizer = BertTokenizer.from_pretrained('pre_model/RoBERTa_zh_L12_PyTorch')
    features = []
    all_labels = []
    for i, (item,_tmp_label) in enumerate(zip(text_data,data_labels)):
      if debug_flag and i>500:
        break
      if i%1000 == 0 and i>0:
        logger.info(i)
      with codecs.open(item,'r','utf-8') as fr_2:
        line = fr_2.read()
        data_json = json.loads(line)
        ocr = data_json['ocr']
        asr = data_json['asr']
        title = data_json['title']
        comment_text = title + '。' + asr[:200] + ocr[-200:]
        comment_text = " ".join(comment_text.split())

        inputs = word_tokenizer(
            comment_text,
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True
        )
        ids = torch.tensor(inputs['input_ids'], dtype=torch.long)
        mask = torch.tensor(inputs['attention_mask'], dtype=torch.long) 
        token_type_ids = torch.tensor(inputs["token_type_ids"], dtype=torch.long) 
        features.append({'ids':ids,'mask':mask,'token_type_ids':token_type_ids})
        all_labels.append(_tmp_label)

        tmp_num = _tmp_label.item()
        if tmp_num in time_result and time_result[tmp_num] >1:
          for _time in range(time_result[tmp_num]-1):
            comment_tokens = word_tokenizer.tokenize(comment_text)
            for k in range(len(comment_tokens)):
              rn = random.randint(0,99)
              if rn >85:
                comment_tokens[k] = "[MASK]"
            tmp_text = word_tokenizer.convert_tokens_to_string(comment_tokens)

            inputs = word_tokenizer(
            tmp_text,
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True
            )
            ids = torch.tensor(inputs['input_ids'], dtype=torch.long)
            mask = torch.tensor(inputs['attention_mask'], dtype=torch.long) 
            token_type_ids = torch.tensor(inputs["token_type_ids"], dtype=torch.long) 
            features.append({'ids':ids,'mask':mask,'token_type_ids':token_type_ids})
            all_labels.append(_tmp_label)
                    
    return features,all_labels


def get_npy_feature_add(data_labels,time_result,npy_feature_path,debug_flag):
  npy_feature = []
  all_labels = []
  for i, (item,_tmp_label) in enumerate(zip(npy_feature_path,data_labels)):
    if i>500 and debug_flag:
      break
    if i%1000 == 0 and i>0:
      logger.info(i)
    try:
      frames = np.load(item)
      frames_select = frames[0,:]
    except:
      frames_select = np.array([0.]*2048)
      logger.info('data error :{}'.format(item))
    features = torch.tensor(frames_select, dtype=torch.float)
    npy_feature.append(features)
    all_labels.append(_tmp_label)

    tmp_num = _tmp_label.item()
    if tmp_num in time_result and time_result[tmp_num] >1:
      try:
          frames = np.load(item)
      except:
          logger.info('data error :{}'.format(item))
          continue
      for _time in range(time_result[tmp_num]-1):
        rn = random.randint(1,10)
        frames_select = frames[rn,:]
        features = torch.tensor(frames_select, dtype=torch.float)
        npy_feature.append(features)
        all_labels.append(_tmp_label)

  return npy_feature,all_labels


def get_data_add_version(data_version,level,data_path,
                          npy_path_dir='/data1/lvzhengwei/git_code/torch_dir/ccks_data/Data/CCKS_dataset/tsn_features_train/',
                          debug_flag=False): 
  seed_random = 5121
  random.seed(seed_random)
  def get_label_add_time():
    label_2_id,id_2_label = get_label_name_dict('ccks/data/{}/{}_label.txt'.format(data_version,level))
    data_path = 'ccks/data/{}/{}_labels_static.json'.format(data_version,level)
    result = {}
    data_json = json.load(codecs.open(data_path,'r'))
    if level == 'level1':
      for key,value in data_json.items():
        if value<1000:
          add_time = int(1000/value) + 1
          result[label_2_id[key]] = add_time

    elif level == 'level2':
      for key,value in data_json.items():
        if value<300:
          add_time = int(200/value) + 1
          result[label_2_id[key]] = add_time
        if value > 1500:
          result[label_2_id[key]] = 1500/value
    return result
  time_result = get_label_add_time()
  npy_data = []
  text_data = []
  data_labels= []
  with codecs.open(data_path,'r','utf-8') as fr:
    for line in fr:
      text_path = line.strip().split()[0]
      label = int(line.strip().split()[1]) 

      ## 进行下采样.
      tmp_num = label
      if tmp_num in time_result and time_result[tmp_num] <1:
        rn = random.randint(0,99)
        keep_data_time = int(time_result[tmp_num]*100)
        if rn >keep_data_time:
          continue

      label = torch.tensor(label, dtype=torch.long)
      data_labels.append(label)
      npy_path = npy_path_dir + text_path.strip().split('/')[-1][:-4] + '.npy'
      npy_data.append(npy_path)
      text_data.append(text_path)

  logger.info('load text data ...')
  text_feature,label_new1 = get_text_feature_add(data_labels,time_result,text_data,debug_flag)
  logger.info('load npy data ...')
  npy_feature,label_new2 = get_npy_feature_add(data_labels,time_result,npy_data,debug_flag)

  return npy_feature,text_feature,label_new2





