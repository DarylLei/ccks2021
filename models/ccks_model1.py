import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ccks.models.NeXtVLAD import SeFusion,SigmodModel,NeXtVLAD
from transformers import BertModel, BertConfig
from ccks.r_drop import compute_kl_loss

## nextvald
class BERTClass(torch.nn.Module):
    def __init__(self,class_num):
        super(BERTClass, self).__init__()
        self.model_name = 'text_model'
        # self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased',cache_dir='data/cache')
        # model_config = BertConfig.from_pretrained('pre_model/RoBERTa_zh_L12_PyTorch')
        # self.bertmodel = BertModel.from_pretrained('pre_model/RoBERTa_zh_L12_PyTorch',config=model_config)
        model_config = BertConfig.from_pretrained('hfl/chinese-roberta-wwm-ext',cache_dir='data/cache')
        self.bertmodel = BertModel.from_pretrained('hfl/chinese-roberta-wwm-ext',config=model_config)
        self.drop_out = torch.nn.Dropout(0.3)

        self.text_model = NeXtVLAD(dim=768, max_frames=400, lamb=2,
                                       num_clusters=128, groups=8)
        input_hidden_dim = int(768 *128*2/8)
        self.fusion_model = SeFusion(input_hidden_dim)

        self.fc_classify = torch.nn.Linear(1024, class_num)
        self.loss = torch.nn.CrossEntropyLoss()
    
    def forward(self, ids, mask, token_type_ids,labels=None):
        outputs = self.bertmodel(ids, attention_mask = mask, token_type_ids = token_type_ids)
        pooled_output = outputs[1]  

        seq_output = outputs[0]  
        seq_output = self.drop_out(seq_output)

        text_feature = self.text_model(seq_output)
        fusion_feature = self.fusion_model([text_feature])

        logits = self.fc_classify(fusion_feature)
        output = {'logits':logits}
        if labels is not None:
          loss = self.loss(logits,labels)
          output['loss'] = loss
        return output

class BERTClassRdrop(torch.nn.Module):
    def __init__(self,class_num):
        super(BERTClassRdrop, self).__init__()
        self.model_name = 'text_bert_rdrop'
        # self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased',cache_dir='data/cache')
        # model_config = BertConfig.from_pretrained('pre_model/RoBERTa_zh_L12_PyTorch')
        # self.bertmodel = BertModel.from_pretrained('pre_model/RoBERTa_zh_L12_PyTorch',config=model_config)
        model_config = BertConfig.from_pretrained('hfl/chinese-roberta-wwm-ext',cache_dir='data/cache')
        self.bertmodel = BertModel.from_pretrained('hfl/chinese-roberta-wwm-ext',config=model_config)
        self.drop_out = torch.nn.Dropout(0.3)
        self.fc_classify = torch.nn.Linear(768, class_num)
        self.loss = torch.nn.CrossEntropyLoss()
    
    def forward(self, ids, mask, token_type_ids,labels=None):
        outputs = self.bertmodel(ids, attention_mask = mask, token_type_ids = token_type_ids)
        outputs2 = self.bertmodel(ids, attention_mask = mask, token_type_ids = token_type_ids)
        pooled_output = outputs[1]         
        pooled_output2 = outputs2[1]         
        pooled_output = self.drop_out(pooled_output)
        pooled_output2 = self.drop_out(pooled_output2)
        logits = self.fc_classify(pooled_output)
        logits2 = self.fc_classify(pooled_output2)

        output = {'logits':logits}
        if labels is not None:
          ce_loss = 0.5 * (self.loss(logits, labels) + self.loss(logits2, labels))
          kl_loss = compute_kl_loss(logits, logits2)
          # carefully choose hyper-parameters
          α = 0.5
          loss = ce_loss + α * kl_loss

          # loss = self.loss(logits,labels)
          output['loss'] = loss
        return output

class BERTLevel2(torch.nn.Module):
    def __init__(self,class_num_1,class_num_2):
        super(BERTLevel2, self).__init__()
        self.model_name = 'level2_bert'
        # self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased',cache_dir='data/cache')
        # model_config = BertConfig.from_pretrained('pre_model/RoBERTa_zh_L12_PyTorch')
        # self.bertmodel = BertModel.from_pretrained('pre_model/RoBERTa_zh_L12_PyTorch',config=model_config)
        model_config = BertConfig.from_pretrained('hfl/chinese-roberta-wwm-ext',cache_dir='data/cache')
        self.bertmodel = BertModel.from_pretrained('hfl/chinese-roberta-wwm-ext',config=model_config)
        self.drop_out = torch.nn.Dropout(0.3)

        self.fc_3 = torch.nn.Linear(class_num_1, class_num_2)

        self.fc_classify_1 = torch.nn.Linear(768, class_num_1)
        self.fc_classify_2 = torch.nn.Linear(768, class_num_2)
        self.loss = torch.nn.CrossEntropyLoss()
    
    def forward(self, ids, mask, token_type_ids,labels1=None,labels2=None):
        outputs = self.bertmodel(ids, attention_mask = mask, token_type_ids = token_type_ids)
        pooled_output = outputs[1]         
        pooled_output = self.drop_out(pooled_output)
        logits_1 = self.fc_classify_1(pooled_output)

        logits_2 = self.fc_classify_2(pooled_output)
        logits_3 = self.fc_3(logits_1)
        logits_f = logits_2 + logits_3

        output = {'logits2':logits_2,'logits1':logits_1}
        if labels1 is not None:
          loss_1 = self.loss(logits_1,labels1)
          loss_2 = self.loss(logits_f,labels2)
          output['loss'] =  0.3 * loss_1 + 0.7 * loss_2
        return output
