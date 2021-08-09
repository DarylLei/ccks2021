import torch.nn as nn
import pretrainedmodels
import torch
import torch.nn.functional as F
from transformers import BertModel, BertConfig
from transformers import RobertaConfig, RobertaModel

def demo():
  print(len(pretrainedmodels.model_names))  # 45
  print(pretrainedmodels.model_names)

  # {'imagenet': {'url': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth', 'input_space': 'RGB', 'input_size': [3, 299, 299], 'input_range': [0, 1], 'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5], 'num_classes': 1000}}
  print(pretrainedmodels.pretrained_settings['inceptionv3'])

  model=pretrainedmodels.__dict__['inceptionv3'](num_classes=1000, pretrained='imagenet')
  # num_ftrs = resnet50.last_linear.in_features
  # resnet50.last_linear = nn.Linear(num_ftrs, 7)

  print(type(model))   
  print(dir(model))   #
  print(model.last_linear)  # Linear(in_features=2048, out_features=1000, bias=True)
  print(model.last_linear.in_features)  # 2048
  print(model.parameters)



class Image_bone(nn.Module):

  def __init__(self,config):
      super(Image_bone,self).__init__()
      self.model=pretrainedmodels.__dict__['inceptionv3'](num_classes=1000, pretrained='imagenet')
      # fine tuning
      dim_feats = self.model.last_linear.in_features # =2048
      nb_classes = config.num_classes
      self.model.last_linear = nn.Linear(dim_feats, nb_classes)

      # from torchvision import models
      # self.model = models.inception_v3(pretrained=True)
      # num_ftrs = self.model.fc.in_features
      # self.model.fc = nn.Linear(num_ftrs, config.num_classes)

  def forward(self,input):
    features = self.model.features(input)    # (N,2048,8,8)
    x = F.avg_pool2d(features, kernel_size=8) # 1 x 1 x 2048
    x = F.dropout(x, training=self.training) # 1 x 1 x 2048
    x = x.view(x.size(0), -1) # 2048
    output = self.model.logits(features)       # (1,1000)
    # output = self.model(input)
    return x, output


class Text_bone(nn.Module):
  def __init__(self,config):
    super(Text_bone,self).__init__()
    LEARNING_RATE = 1e-05
    pre_model_path = 'pre_model/RoBERTa_zh_L12_PyTorch'
    # model_config = BertConfig.from_pretrained(pre_model_path)
    # self.l1 = BertModel.from_pretrained(pre_model_path,config=model_config)
    model_config = BertConfig.from_pretrained('hfl/chinese-roberta-wwm-ext',cache_dir='data/cache')
    self.l1 = BertModel.from_pretrained('hfl/chinese-roberta-wwm-ext',config=model_config)
    self.l2 = torch.nn.Dropout(0.5)

    
  def forward(self, ids, mask, token_type_ids):
      outputs = self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids)
      pooled_output = outputs[1]         
      output = self.l2(pooled_output)
      return output



if __name__=='__main__':
  demo()
  pass



