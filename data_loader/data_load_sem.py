import numpy as np
import codecs,glob,json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from transformers.utils.dummy_pt_objects import DebertaForMaskedLM
from logger import logger

def get_sem_label_name_dict(path):
    with codecs.open(path,'r','utf-8') as fr:
        label_2_id = json.load(fr)
    id_2_label = {value:key for key,value in label_2_id.items()}
    return label_2_id,id_2_label

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text_feature = self.data[index]['text_feature']
        labels = self.data[index]['labels']
        tokens = self.data[index]['tokens']
        return {
            'text_feature': text_feature,
            'labels':labels,
            'tokens':tokens
        }

class CustomTestDataset(Dataset):
    def __init__(self, data):
        self.text_feature,self.sample_ids = data

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        text_feature = self.text_feature[index]
        _id = self.sample_ids[index]
        return {
            'text_feature': text_feature,
            'video_id':_id
        }

def _read(data_file, label_map_file):
    with open(label_map_file, "r") as inf:
        tag2label = json.load(inf)
    with open(data_file, 'r', encoding='utf-8') as inf:
        for line in inf:
            line_stripped = line.strip().split('\t')
            assert len(line_stripped) == 2
            tokens = line_stripped[0].split("\002")
            tags = line_stripped[1].split("\002")
            labels = [tag2label[tag] for tag in tags]
            yield {"tokens": tokens, "labels": labels}


def get_data(data_path,data_version,debug_flag=False,):
    label_map_file = 'data/sem_tag/{}/label_map.json'.format(data_version)
    word_tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
    max_length = 500
    data = []
    data_iter = _read(data_path,label_map_file)
    for i, item in enumerate(data_iter):
      if debug_flag and i>500:
        break
      
      ids = word_tokenizer.convert_tokens_to_ids(item['tokens'])
      ids = [101] + ids + [102]
      tokens = ['CLS'] + item['tokens'] + ["SEP"]
      if len(ids) > max_length:
        pad_ids = ids[:max_length-1] + [102]
        tokens = tokens[:max_length-1] + ["SEP"]
        attention_mask = [1] * max_length
        token_type_ids = [0] * max_length
        labels = [2] + item['labels'][:max_length-2] + [2]
        
      else:
        pad_ids = ids + [0]*(max_length-len(ids))
        tokens = tokens + ['PAD']*(max_length-len(ids))
        attention_mask = [1] * len(ids) + [0]*(max_length-len(ids))
        token_type_ids = [0] * max_length
        labels = [2] + item['labels'] + [2] + [2]*(max_length-len(ids))
      # inputs = word_tokenizer(
      #     '北京天哪么么发生',
      #     max_length=4,
      #     padding='max_length',
      #     truncation=True
      # )
      pad_ids = torch.tensor(pad_ids, dtype=torch.long)
      mask = torch.tensor(attention_mask, dtype=torch.bool) 
      token_type_ids = torch.tensor(token_type_ids, dtype=torch.long) 
      labels = torch.tensor(labels, dtype=torch.long) 
      data.append({'text_feature':{'ids':pad_ids,'mask':mask,'token_type_ids':token_type_ids},
                    'labels':labels,
                    'tokens':tokens})
    return data


def get_text_feature(text_data,debug_flag,MAX_LEN=400):
    word_tokenizer = BertTokenizer.from_pretrained('pre_model/RoBERTa_zh_L12_PyTorch')
    features = []
    for i, item in enumerate(text_data):
      if debug_flag and i>100:
        break
      if i%1000 == 0 and i>0:
        logger.info(i)
      with codecs.open(item,'r','utf-8') as fr_2:
        line = fr_2.read()
        data_json = json.loads(line)
        ocr = data_json['ocr']
        asr = data_json['asr']
        title = data_json['title']
        comment_text = title + '。' + asr
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


def build_dataloader(config,debug_flag,data_version):
   
  TRAIN_BATCH_SIZE = config.train_batch_size
  VALID_BATCH_SIZE = config.valid_batch_size

  # train_data_path = 'ccks/data/train.tsv'
  # valid_data_path = 'ccks/data/val.tsv'
  train_data_path = 'data/sem_tag/{}/train.tsv'.format(data_version)
  valid_data_path = 'data/sem_tag/{}/val.tsv'.format(data_version)


  logger.info(train_data_path)
  train_data = get_data(train_data_path,data_version,debug_flag)
  valid_data = get_data(valid_data_path,data_version,debug_flag)

  logger.info("TRAIN Dataset: {}".format(len(train_data)))
  logger.info("VALID Dataset: {}".format(len(valid_data)))

  train_set = CustomDataset(train_data)
  valid_set = CustomDataset(valid_data)

  train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }
  valid_params = {'batch_size': VALID_BATCH_SIZE,
                  'shuffle': False,
                  'num_workers': 0
                  }

  train_loader = DataLoader(train_set, **train_params)
  valid_loader = DataLoader(valid_set, **valid_params)

  return train_loader,valid_loader

def get_predict_data(debug_flag,data_version):
  test_data_path = 'data/sem_tag/{}/test.tsv'.format(data_version)
  test_data = get_data(test_data_path,data_version,debug_flag)
  logger.info("Test Dataset: {}".format(len(test_data)))
  test_set = CustomDataset(test_data)
  test_params = {'batch_size': 24,
                  'shuffle': False,
                  'num_workers': 2
                  }
  test_loader = DataLoader(test_set, **test_params)
  logger.info('predict data batch: {},data_version :{}'.format(len(test_loader),data_version))
  return test_loader

def get_val_data(debug_flag,data_version):
  test_data_path = 'data/sem_tag/{}/val.tsv'.format(data_version)
  test_data = get_data(test_data_path,data_version,debug_flag)
  logger.info("val Dataset: {}".format(len(test_data)))
  test_set = CustomDataset(test_data)
  test_params = {'batch_size': 16,
                  'shuffle': False,
                  'num_workers': 2
                  }
  test_loader = DataLoader(test_set, **test_params)

  return test_loader


