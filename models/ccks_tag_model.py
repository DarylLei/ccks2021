import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from transformers import BertModel, BertConfig
from models.NeXtVLAD import SeFusion
from models.r_drop import compute_kl_loss

class BERTMulti(torch.nn.Module):
    def __init__(self,class_num):
        super(BERTMulti, self).__init__()
        self.model_name = 'bert_multi'
        # self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased',cache_dir='data/cache')
        # model_config = BertConfig.from_pretrained('pre_model/RoBERTa_zh_L12_PyTorch')
        # self.l1 = BertModel.from_pretrained('pre_model/RoBERTa_zh_L12_PyTorch',config=model_config)
        model_config = BertConfig.from_pretrained('hfl/chinese-roberta-wwm-ext',cache_dir='data/cache')
        self.bertmodel = BertModel.from_pretrained('hfl/chinese-roberta-wwm-ext')
        # self.bertmodel = BertModel.from_pretrained('hfl/chinese-roberta-wwm-ext-large')
        # self.dropout = torch.nn.Dropout(0.3)
        self.fusion_model = SeFusion(768+2048)
        self.fc_classify = torch.nn.Linear(2048, class_num)
        self.loss = torch.nn.CrossEntropyLoss()
    
    def forward(self, ids, mask, token_type_ids,npy_feature,labels=None):
        outputs = self.bertmodel(ids, attention_mask = mask, token_type_ids = token_type_ids)
        pooled_output = outputs[1]
        fusion_feature = self.fusion_model([pooled_output,npy_feature])         
        # output_2 = self.l2(pooled_output)
        logits = self.fc_classify(fusion_feature)
        output = {'logits':logits}
        if labels is not None:
          loss = self.loss(logits,labels)
          output['loss'] = loss
        return output

class BERTMultiRDrop(torch.nn.Module):
    def __init__(self,class_num):
        super(BERTMultiRDrop, self).__init__()
        self.model_name = 'bert_multi_rdrop'
        self.inter_model = BERTMulti(class_num)
        self.loss = torch.nn.CrossEntropyLoss()
    
    def forward(self, ids, mask, token_type_ids,npy_feature,labels=None):
        output1 = self.inter_model(ids, mask, token_type_ids,npy_feature,labels)
        output2 = self.inter_model(ids, mask, token_type_ids,npy_feature,labels)

        logits1 = output1['logits']
        logits2 = output2['logits']

        output = {'logits':logits1}
        if labels is not None:
          ce_loss = 0.5 * (self.loss(logits1, labels) + self.loss(logits2, labels))
          kl_loss = compute_kl_loss(logits1, logits2)
          # carefully choose hyper-parameters
          α = 0.5
          loss = ce_loss + α * kl_loss
          output['loss'] = loss
        return output



class Model(nn.Module):
    def __init__(self, num_classes):
        super(Model, self).__init__()
        embedding_dim = 2048
        hidden_dim = 512
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, bidirectional=True,
                              batch_first=True,
                              dropout=0.5)
        self.fc = nn.Linear(hidden_dim* 2, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.loss = torch.nn.CrossEntropyLoss()

        #x,query：[batch, seq_len, hidden_dim*2]
    def attention_net(self, x, query, mask=None):      #软性注意力机制（key=value=x）
        d_k = query.size(-1)                                              #d_k为query的维度
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)  #打分机制  scores:[batch, seq_len, seq_len]
        p_attn = F.softmax(scores, dim = -1)                              #对最后一个维度归一化得分
        context = torch.matmul(p_attn, x).sum(1)       #对权重化的x求和，[batch, seq_len, hidden_dim*2]->[batch, hidden_dim*2]
        return context, p_attn

    def forward(self, x, labels=None):
        embedding = self.dropout(x)       #[seq_len, batch, embedding_dim]

        # output: [seq_len, batch, hidden_dim*2]     hidden/cell: [n_layers*2, batch, hidden_dim]
        output, (final_hidden_state, final_cell_state) = self.lstm(embedding)
        # output = output.permute(1, 0, 2)                  #[batch, seq_len, hidden_dim*2]

        query = self.dropout(output)
        attn_output, attention = self.attention_net(output, query)       #和LSTM的不同就在于这一句
        logits = self.fc(attn_output)
        
        output = {'logits':logits}
        if labels is not None:
          loss = self.loss(logits,labels)
          output['loss'] = loss
        return output

class BiLSTM_Attention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers):

        super(BiLSTM_Attention, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=True, dropout=0.5)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(0.5)

    #x,query：[batch, seq_len, hidden_dim*2]
    def attention_net(self, x, query, mask=None):      #软性注意力机制（key=value=x）

        d_k = query.size(-1)                                              #d_k为query的维度
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)  #打分机制  scores:[batch, seq_len, seq_len]

        p_attn = F.softmax(scores, dim = -1)                              #对最后一个维度归一化得分
        context = torch.matmul(p_attn, x).sum(1)       #对权重化的x求和，[batch, seq_len, hidden_dim*2]->[batch, hidden_dim*2]
        return context, p_attn


    def forward(self, x):
        embedding = self.dropout(self.embedding(x))       #[seq_len, batch, embedding_dim]

        # output: [seq_len, batch, hidden_dim*2]     hidden/cell: [n_layers*2, batch, hidden_dim]
        output, (final_hidden_state, final_cell_state) = self.rnn(embedding)
        output = output.permute(1, 0, 2)                  #[batch, seq_len, hidden_dim*2]

        query = self.dropout(output)
        attn_output, attention = self.attention_net(output, query)       #和LSTM的不同就在于这一句
        logit = self.fc(attn_output)
        return logit