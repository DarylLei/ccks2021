import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig
from models.r_drop import compute_kl_loss
from torchcrf import CRF

from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)


class BertSoftmax(torch.nn.Module):
    def __init__(self,class_num):
        super(BertSoftmax, self).__init__()
        model_config = BertConfig.from_pretrained('hfl/chinese-roberta-wwm-ext',cache_dir='data/cache')
        self.bertmodel = BertModel.from_pretrained('hfl/chinese-roberta-wwm-ext',config=model_config)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, class_num)
        self.loss = torch.nn.CrossEntropyLoss()
        self.num_labels = class_num
    
    def forward(self, ids, mask, token_type_ids,labels=None):
        outputs = self.bertmodel(ids, attention_mask = mask, token_type_ids = token_type_ids)
        # pooled_output = outputs[1]         
        # output_2 = self.dropout(pooled_output)
        # output = self.classifier(output_2)

        sequence_output = outputs[0]
        # padded_sequence_output = self.dropout(sequence_output)
        padded_sequence_output = sequence_output
        # 得到判别值
        logits = self.classifier(padded_sequence_output)
        outputs = {'logits':logits}
        if labels is not None:
            loss_mask = labels.gt(2)
            # Only keep active parts of the loss
            if loss_mask is not None:
                # 只留下label存在的位置计算loss
                # active_loss = loss_mask.view(-1) == 1
                # active_loss = labels.view(-1) != 2
                active_loss = mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = self.loss(active_logits, active_labels)
            else:
                loss = self.loss(logits.view(-1, self.num_labels), labels.view(-1))
            outputs['loss'] = loss      
        return outputs

class BertCRF(torch.nn.Module):
    def __init__(self, class_num):
        super(BertCRF, self).__init__()
        self.model_name = 'BertCRF'
        model_config = BertConfig.from_pretrained('hfl/chinese-roberta-wwm-ext',cache_dir='data/cache')
        self.bertmodel = BertModel.from_pretrained('hfl/chinese-roberta-wwm-ext',config=model_config)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, class_num)
        self.loss = torch.nn.CrossEntropyLoss()
        self.num_labels = class_num
        self.crf = CRF(class_num,batch_first=True)

    def forward(self, ids, mask, token_type_ids,labels=None):
        outputs = self.bertmodel(ids, attention_mask = mask, token_type_ids = token_type_ids)
        sequence_output = outputs[0]
        padded_sequence_output = self.dropout(sequence_output)
        # 得到判别值
        logits = self.classifier(padded_sequence_output)
        outputs = {'logits':logits}
        if labels is not None:
            # loss_mask = mask.view(-1) == 1
            # mask = mask.permute(1,0)
            # logits = logits.permute(1,0,2)
            # labels = labels.permute(1,0)
            loss = self.crf(logits, labels, mask)* (-1) 
            # outputs = (loss,) + outputs
            outputs['loss'] = loss

        return outputs

class BertCRFRdrop(torch.nn.Module):
    def __init__(self,class_num):
        super(BertCRFRdrop, self).__init__()
        self.model_name = 'BertCRFRdrop'
        model_config = BertConfig.from_pretrained('hfl/chinese-roberta-wwm-ext',cache_dir='data/cache')
        self.bertmodel = BertModel.from_pretrained('hfl/chinese-roberta-wwm-ext',config=model_config)
        self.dropout = torch.nn.Dropout(0.3)
        self.fc_classify = torch.nn.Linear(768, class_num)
        self.loss = torch.nn.CrossEntropyLoss()
        self.num_labels = class_num
        self.crf = CRF(class_num,batch_first=True)
    
    def forward(self, ids, mask, token_type_ids,labels=None):
        outputs = self.bertmodel(ids, attention_mask = mask, token_type_ids = token_type_ids)
        outputs2 = self.bertmodel(ids, attention_mask = mask, token_type_ids = token_type_ids)
        sequence_output = outputs[0]         
        sequence_output2 = outputs2[0]         
        sequence_output = self.dropout(sequence_output)
        sequence_output2 = self.dropout(sequence_output2)

        logits = self.fc_classify(sequence_output)
        logits2 = self.fc_classify(sequence_output2)

        output = {'logits':logits}
        if labels is not None:
          loss = self.crf(logits, labels, mask)* (-1) 
          ce_loss = 0.5 * (self.crf(logits, labels, mask)* (-1) + self.crf(logits2, labels, mask)* (-1))
          tmp_mask = mask.eq(0)
          tmp_mask = tmp_mask.unsqueeze(-1)
          kl_loss = compute_kl_loss(logits, logits2,pad_mask=tmp_mask)
          # carefully choose hyper-parameters
          α = 0.5
          loss = ce_loss + α * kl_loss

          # loss = self.loss(logits,labels)
          output['loss'] = loss
        return output

class GenModel(torch.nn.Module):
    def __init__(self,class_num):
        super(GenModel, self).__init__()
        config = AutoConfig.from_pretrained('t5-small')
        self.tokenizer = AutoTokenizer.from_pretrained('t5-small')
        self.model = AutoModelForSeq2SeqLM.from_pretrained('t5-small')

        self.model.resize_token_embeddings(len(tokenizer))

        if self.model.config.decoder_start_token_id is None:
            raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    
    def forward(self, ids, mask, token_type_ids,labels=None):
        prefix = "summarize: "
        max_target_length = 500
        padding = 'max_length'
        ignore_pad_token_for_loss = True
        def preprocess_function(examples):
            inputs = examples['text']
            targets = examples['summary']
            inputs = [prefix + inp for inp in inputs]
            model_inputs = self.tokenizer(inputs, max_length=max_target_length, padding=padding, truncation=True)

            # Setup the tokenizer for targets
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

            # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
            # padding in the loss.
            if padding == "max_length" and ignore_pad_token_for_loss:
                labels["input_ids"] = [
                    [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
                ]
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
            
        outputs = {'logits':logits}
        if labels is not None:
            loss = self.loss(logits.view(-1, self.num_labels), labels.view(-1))
            outputs['loss'] = loss      
        return outputs


