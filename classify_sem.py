import codecs
import json
from operator import le
import sys,os
sys.path.append('.')
os.environ['CUDA_VISIBLE_DEVICES']='0,1' 
import numpy as np
import torch
from torch import cuda
from data_loader.data_load_sem import build_dataloader,get_predict_data,get_sem_label_name_dict,get_val_data
from metrics import f1_score,bad_case, f1_score_ac,get_entity_name,modify_with_rule
from transformers import BertModel, BertConfig
from logger import logger
from torch.utils.tensorboard import SummaryWriter 
import time
from classify_tag_mul import get_parameter_number
import torch.nn.functional as F
from sklearn.metrics import classification_report
import torch.nn as nn
from models import ccks_model2


class config_tmp:
    def __init__(self,data_version) -> None:
        self.train_batch_size = 4
        self.valid_batch_size =4
        self.num_epochs = 5
        self.device = 'cuda' if cuda.is_available() else 'cpu'
        self.model_name = 'semtag'
        self.data_version = data_version
    def create_path(self):
        self.save_path = 'ccks/output_'+ self.data_version + '/' + self.model_name + '/saved_model/'
        self.log_path = 'ccks/output_'+ self.data_version + '/' + self.model_name + '/log_board/'
        for path in [self.save_path,self.log_path]:
          if not os.path.exists(path):
            os.makedirs(path)
            logger.info('create logger path:{}'.format(path))


def validation_crf(config, model, dev_iter,data_version):
    model.eval()
    loss_total = 0
    pred_tags = []
    true_tags = []
    pred_tags_label = []
    true_tags_label = []
    sample_tokens = []
    device = config.device
    label_2_id,id_2_label= get_sem_label_name_dict('data/sem_tag/{}/label_map.json'.format(data_version))
    with torch.no_grad():
        for _, data in enumerate(dev_iter):
            text_feature = data['text_feature']
            _tokens = data['tokens']
            ids, mask, token_type_ids = text_feature.values()
            ids, mask, token_type_ids = ids.to(device), mask.to(device), token_type_ids.to(device)
            labels = data['labels'].to(device)
            output = model(ids , mask, token_type_ids,labels)
            logits = output['logits']
            loss = output['loss']
            loss_total += loss.mean()
            pre_index = model.module.crf.decode(logits,mask=mask)
            labels = labels.cpu().detach().numpy()
            _tmp = [[id_2_label.get(idx) for idx in indices] for indices in pre_index]
            _tmp = [indices + ['O'] *(500-len(indices)) for indices in _tmp]
            pred_tags.extend(_tmp)
            # true_tags.extend([[id_2_label.get(idx) if idx != -1 else 'O' for idx in indices] for indices in labels])
            true_tags.extend([[id_2_label.get(idx) for idx in indices] for indices in labels])
            # tt = list(zip(*_tokens)) 
            for _sub in list(zip(*_tokens)):
                tmp_list = []
                for _item in _sub:
                    if _item == 'PAD':
                        break
                    tmp_list.append(_item)
                sample_tokens.append(tmp_list)
                    
            # sample_tokens.extend([indices for indices in zip(*_tokens)])

    # logging loss, f1 and report
    # metric_result = classification_report(true_tags, pred_tags, zero_division=0, output_dict=True)
    metrics = {}
    mode = 'dev'
    if mode == 'dev':
        p,r,f1 = f1_score(true_tags, pred_tags, mode)
        logger.info('p: {:.4}  r: {:.4} f1: {:.4} '.format(p,r,f1))
        p_a,r_a,f1_ac = f1_score_ac(pred_tags, sample_tokens,data_version)
        logger.info('p_ac: {:.4}  r_ac: {:.4} f1_ac: {:.4} '.format(p_a,r_a,f1_ac))
        metrics['f1'] = f1
        metrics['f1_ac'] = f1_ac
    else:
        bad_case(true_tags, pred_tags, sample_tokens)
        f1_labels, f1 = f1_score(true_tags, pred_tags, mode)
        metrics['f1_labels'] = f1_labels
        metrics['f1'] = f1
    metrics['loss'] = float(loss_total / len(dev_iter))
    model.train()
    return metrics

def validation(config, model, dev_iter):
    model.eval()
    loss_total = 0
    pred_tags = []
    true_tags = []
    pred_tags_label = []
    true_tags_label = []
    sample_tokens = []
    device = config.device
    with torch.no_grad():
        for _, data in enumerate(dev_iter):
            text_feature = data['text_feature']
            _tokens = data['tokens']
            ids, mask, token_type_ids = text_feature.values()
            ids, mask, token_type_ids = ids.to(device), mask.to(device), token_type_ids.to(device)
            labels = data['labels'].to(device)
            output = model(ids , mask, token_type_ids,labels)
            logits = output['logits']
            loss = output['loss']
            loss_total += loss.mean()
            pred_probility = F.softmax(logits,dim=2)
            pre_index=torch.argmax(pred_probility,dim=2).cpu().numpy()
            labels = labels.cpu().detach().numpy()

            pred_tags.extend([[id_2_label.get(idx) for idx in indices] for indices in pre_index])
            # true_tags.extend([[id_2_label.get(idx) if idx != -1 else 'O' for idx in indices] for indices in labels])
            true_tags.extend([[id_2_label.get(idx) for idx in indices] for indices in labels])
            # tt = list(zip(*_tokens)) 
            for _sub in list(zip(*_tokens)):
                tmp_list = []
                for _item in _sub:
                    if _item == 'PAD':
                        break
                    tmp_list.append(_item)
                sample_tokens.append(tmp_list)
                    
            # sample_tokens.extend([indices for indices in zip(*_tokens)])

    # logging loss, f1 and report
    # metric_result = classification_report(true_tags, pred_tags, zero_division=0, output_dict=True)
    metrics = {}
    mode = 'dev'
    if mode == 'dev':
        p,r,f1 = f1_score(true_tags, pred_tags, mode)
        logger.info('p: {:.4}  r: {:.4} f1: {:.4} '.format(p,r,f1))
        p_a,r_a,f1_ac = f1_score_ac(pred_tags, sample_tokens)
        logger.info('p_ac: {:.4}  r_ac: {:.4} f1_ac: {:.4} '.format(p_a,r_a,f1_ac))
        metrics['f1'] = f1
        metrics['f1_ac'] = f1_ac
    else:
        bad_case(true_tags, pred_tags, sample_tokens)
        f1_labels, f1 = f1_score(true_tags, pred_tags, mode)
        metrics['f1_labels'] = f1_labels
        metrics['f1'] = f1
    metrics['loss'] = float(loss_total / len(dev_iter))
    model.train()
    return metrics

def run_train(debug_flag,data_vesion):
    logger.info("---#### start training ####---")
    logger.info("---#### main_npy.py ####---")
    seed_random = 5124
    np.random.seed(seed_random)
    torch.manual_seed(seed_random)
    torch.cuda.manual_seed_all(seed_random)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样
    logger.info('随机种子：{}'.format(seed_random))

    class_num = 3
    best_dev_acc = 0.379
    # model = ccks_model2.BertSoftmax(class_num)
    # model = ccks_model2.BertCRF(class_num)
    model = ccks_model2.BertCRFRdrop(class_num)
    config = config_tmp(data_vesion)
    config.model_name = model.model_name
    config.create_path()
    
    if torch.cuda.device_count() > 1:
      logger.info("Let's use {} GPUs!".format(torch.cuda.device_count()))
      config.train_batch_size = config.train_batch_size * int(torch.cuda.device_count())
      config.valid_batch_size = config.valid_batch_size * int(torch.cuda.device_count())
      # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
      model = nn.DataParallel(model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info('use device:{}'.format(device))
    model.to(device)

    train_iter, dev_iter = build_dataloader(config,debug_flag,data_vesion)
    get_parameter_number(model)
    # print(model.parameters)

    model.train()
    # optimizer = torch.optim.Adam([  {'params':model.bertmodel.parameters(),'lr':1e-5},
    #                                 {'params':model.classifier.parameters()},       
    #                                     ], lr=1e-3)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    total_batch = 0  # 记录进行到多少batch
    dev_loss,train_acc,dev_acc = 0.0,0.0,0.0
    dev_acc_list = []
    dev_acc_list = []
    true_f1_list = []
    if debug_flag:
      best_dev_acc = 0.
      val_step = 80
    else:
      val_step = 1600
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    for epoch in range(config.num_epochs):
        logger.info('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        # scheduler.step() # 学习率衰减
        for i, data in enumerate(train_iter):
            text_feature = data['text_feature']
            ids, mask, token_type_ids = text_feature.values()
            ids, mask, token_type_ids = ids.to(device), mask.to(device), token_type_ids.to(device)
            labels = data['labels'].to(device)
            output = model(ids , mask, token_type_ids,labels)
            logits = output['logits']
            loss = output['loss'].mean()
            model.zero_grad()
            loss.backward()
            optimizer.step()
            if total_batch % 40 == 0 and total_batch > 0:
                # 每多少轮输出在训练集和验证集上的效果
                # with torch.no_grad():
                #     # pred = torch.sigmoid(logits).cpu().numpy()
                #     pred = F.softmax(logits,dim=1)
                #     max_index=torch.argmax(pred,dim=1).cpu().numpy()
                #     label = labels.cpu().numpy()
                # metric_result = classification_report(label, max_index,output_dict=True)
                # train_acc = metric_result['accuracy']
                logger.info('Iter:{}, epoch:{}/{}  train_loss:{:.4}'
                            .format(total_batch, epoch+1, config.num_epochs, loss.item()))
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                # writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)

            if total_batch %val_step == 0 and total_batch > 0 :
                  logger.info('----#### eval ####----')
                #   output = validation(config, model, dev_iter)
                  output = validation_crf(config, model, dev_iter,data_version)
                  dev_acc, dev_acc_true, dev_loss = output['f1'], output['f1_ac'], output['loss']
                  logger.info('Val loss: {:.4}  val_acc: {:.4} val_acc_true: {:.4} '.format(dev_loss,dev_acc,dev_acc_true))
                  dev_acc_list.append({"iter":total_batch,'dev_acc':dev_acc})
                  true_f1_list.append({"iter":total_batch,'true_f1':dev_acc_true})
                  if dev_acc_true > best_dev_acc:
                      best_dev_acc = dev_acc_true
                      ## 仅保存模型参数
                    #   torch.save(model.state_dict(), config.save_path +'Iter-{}-acc-{:.4}.pth'.format(total_batch,best_dev_acc))
                      ## 保存整个模型
                      torch.save(model, config.save_path +'Iter-{}-acc-{:.4}.pth'.format(total_batch,best_dev_acc))
                
            total_batch += 1
    writer.close()
    dev_acc_list = sorted(dev_acc_list,key=lambda x:x['dev_acc'],reverse=True)
    true_f1_list = sorted(true_f1_list,key=lambda x:x['true_f1'],reverse=True)
    show_line = ''
    for item in dev_acc_list[:3]:
            show_line += '   iter:{}  dev_acc:{:.4}\n'.format(item['iter'],item['dev_acc'])
    logger.info('dev top3 result:\n{}'.format(show_line))
    show_line = ''
    for item in true_f1_list[:3]:
            show_line += '   iter:{}  true_f1:{:.4}\n'.format(item['iter'],item['true_f1'])
    logger.info('dev top3 result:\n{}'.format(show_line))

def run_predict(name,test_data):
    label_2_id,id_2_label= get_label_name_dict('data/sem_tag/{}/label_map.json'.format(data_version))
    model = torch.load('ccks/output/semtag/saved_model/' + name) 
    ## 判断是单卡还是多卡模型
    if isinstance(model,torch.nn.DataParallel) and torch.cuda.device_count() == 1:
	    model = model.module

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    pre_results = []
    for i, data in enumerate(test_data):
        if i%100 == 0 and i>0:
            print(i)
        model.eval()
        with torch.no_grad():
            text_feature = data['text_feature']
            _tokens = data['tokens']
            ids, mask, token_type_ids = text_feature.values()
            ids, mask, token_type_ids = ids.to(device), mask.to(device), token_type_ids.to(device)
            output = model(ids , mask, token_type_ids)
            logits = output['logits']
            pred_probility = F.softmax(logits,dim=2)
            pre_index=torch.argmax(pred_probility,dim=2).cpu().numpy()

            # pred_tags.extend([[id_2_label.get(idx) for idx in indices] for indices in pre_index])
            pred_tags = [[id_2_label.get(idx) for idx in indices] for indices in pre_index]
            for i, _sub in enumerate(list(zip(*_tokens))):
                tmp_token = []
                for _item in _sub:
                    if _item == 'PAD':
                        break
                    tmp_token.append(_item)
                tmp_pre_tag = pred_tags[i][:len(tmp_token)]
                tmp_pre_name = get_entity_name(tmp_pre_tag,tmp_token)
                tmp_pre_name = [ ''.join(_item[0]) for _item in tmp_pre_name]
                pre_results.append(tmp_pre_name)
                
    # 保存结果
    with open("ccks/data/sem_tag_pre/add_asr/test_nids.txt", "r") as inf:
        id_lines =[ _item.strip().split('\t')[0] for _item in inf.readlines()]
    nid2ents = {}
    for entities, nid in zip(pre_results, id_lines):
        nid2ents[nid.strip()] = entities

    path_save = 'ccks/output/result/sem_tag/'
    if not os.path.exists(path_save):
        os.makedirs(path_save)
        logger.info('create logger path:{}'.format(path_save))
    with codecs.open(path_save + name + '.json','w',"utf-8") as f:
        json.dump(nid2ents, f, ensure_ascii=False, indent=4)
    print('predict finished !')

def run_predict_crf(name,test_data,data_version):
    model = torch.load('ccks/output_{}/BertCRFRdrop/saved_model/'.format(data_version) + name) 
    logger.info("Let's use {} GPUs!".format(torch.cuda.device_count()))
    ## 判断是单卡还是多卡模型
    if isinstance(model,torch.nn.DataParallel) and torch.cuda.device_count() == 1:
	    model = model.module
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info('use device:{}'.format(device))
    model.to(device)
    label_2_id,id_2_label= get_sem_label_name_dict('data/sem_tag/{}/label_map.json'.format(data_version))
    pre_results = []
    for i, data in enumerate(test_data):
        if i%100 == 0 and i>0:
            print(i)
        model.eval()
        with torch.no_grad():
            text_feature = data['text_feature']
            _tokens = data['tokens']
            ids, mask, token_type_ids = text_feature.values()
            ids, mask, token_type_ids = ids.to(device), mask.to(device), token_type_ids.to(device)
            output = model(ids , mask, token_type_ids)
            logits = output['logits']
            pre_index = model.module.crf.decode(logits,mask=mask)
            pred_tags = [[id_2_label.get(idx) for idx in indices] for indices in pre_index]
            for i, _sub in enumerate(list(zip(*_tokens))):
                tmp_token = []
                for _item in _sub:
                    if _item == 'PAD':
                        break
                    tmp_token.append(_item)
                tmp_pre_tag = pred_tags[i][:len(tmp_token)]
                tmp_pre_name = get_entity_name(tmp_pre_tag,tmp_token)
                tmp_pre_name = [ ''.join(_item[0]) for _item in tmp_pre_name]
                pre_results.append(tmp_pre_name)
                
    # 保存结果
 
    with open("data/sem_tag/{}/test_nids.txt".format(data_version), "r") as inf:
    # with open("ccks/data/sem_tag_pre/{}/val_nids.txt".format(tmp_version), "r") as inf:
        id_lines =[ _item.strip().split('\t')[0] for _item in inf.readlines()]
    nid2ents = {}
    for entities, nid in zip(pre_results, id_lines):
        nid2ents[nid.strip()] = entities

    path_save = 'ccks/output_{}/result/sem_tag/'.format(data_version)
    # path_save = 'ccks/ensem_test_{}/result/sem_tag/'.format(data_version)
    if not os.path.exists(path_save):
        os.makedirs(path_save)
        logger.info('create logger path:{}'.format(path_save))
    with codecs.open(path_save + name + '.json','w',"utf-8") as f:
        json.dump(nid2ents, f, ensure_ascii=False, indent=4)
    print('predict finished !')


def run_dev(model_name,debug_flag,data_version):
    import codecs,json
    name = model_name[0]
    model = torch.load('ccks/output_{}/BertCRFRdrop/saved_model/'.format(data_version) + name) 
    logger.info("Let's use {} GPUs!".format(torch.cuda.device_count()))
    ## 判断是单卡还是多卡模型
    if isinstance(model,torch.nn.DataParallel) and torch.cuda.device_count() == 1:
	    model = model.module
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info('use device:{}'.format(device))
    model.to(device)
    model.eval()
    label_2_id,id_2_label= get_sem_label_name_dict('data/sem_tag/{}/label_map.json'.format(data_version))
    val_data = get_val_data(debug_flag,data_version)
    pred_tags = []
    true_tags = []
    sample_tokens = []
    with torch.no_grad():
        for i, data in enumerate(val_data):
            if i%100 == 0 and i>0:
                print(i)
            text_feature = data['text_feature']
            _tokens = data['tokens']
            ids, mask, token_type_ids = text_feature.values()
            ids, mask, token_type_ids = ids.to(device), mask.to(device), token_type_ids.to(device)
            labels = data['labels'].to(device)
            output = model(ids , mask, token_type_ids,labels)
            logits = output['logits']
            pre_index = model.module.crf.decode(logits,mask=mask)
            labels = labels.cpu().detach().numpy()
            _tmp = [[id_2_label.get(idx) for idx in indices] for indices in pre_index]
            _tmp = [indices + ['O'] *(500-len(indices)) for indices in _tmp]
            pred_tags.extend(_tmp)
            # true_tags.extend([[id_2_label.get(idx) if idx != -1 else 'O' for idx in indices] for indices in labels])
            true_tags.extend([[id_2_label.get(idx) for idx in indices] for indices in labels])
            # tt = list(zip(*_tokens)) 
            for _sub in list(zip(*_tokens)):
                tmp_list = []
                for _item in _sub:
                    if _item == 'PAD':
                        break
                    tmp_list.append(_item)
                sample_tokens.append(tmp_list)
                
    
    ## 矫正结果
    # p_a,r_a,f1_ac = modify_with_rule(pred_tags, sample_tokens)
    # logger.info('rule_modify -- p_ac: {:.4}  r_ac: {:.4} f1_ac: {:.4} '.format(p_a,r_a,f1_ac))
    # 模型结果
    metrics = {}
    p,r,f1 = f1_score(true_tags, pred_tags, 'dev')
    logger.info('p: {:.4}  r: {:.4} f1: {:.4} '.format(p,r,f1))
    p_a,r_a,f1_ac = f1_score_ac(pred_tags, sample_tokens,data_version)
    logger.info('p_ac: {:.4}  r_ac: {:.4} f1_ac: {:.4} '.format(p_a,r_a,f1_ac))
    metrics['f1'] = f1
    metrics['f1_ac'] = f1_ac
    config_case_dir = 'ccks/output_{}/case/'.format(data_version)
    bad_case(true_tags, pred_tags, sample_tokens,config_case_dir)
    logger.info('predict finished !')




if __name__=='__main__':
    debug_flag = True
    data_version = 'add_asr5'
    logger.info('data_version:{}'.format(data_version))
    # run_train(debug_flag,data_version)
    
    # model_name_rdrop1 = ['Iter-17600-acc-0.3928.pth','Iter-8000-acc-0.3913.pth','Iter-8000-acc-0.3914.pth']
    # model_name_rdrop = ['Iter-16000-acc-0.3924.pth','Iter-11200-acc-0.386.pth']  #  data4
    # model_name_rdrop = ['Iter-8000-acc-0.3923.pth']  #  data4  2
    # model_name_rdrop2 = ['Iter-8000-acc-0.3845.pth','Iter-6400-acc-0.3815.pth']  #  data3
   
    # model_name1 = ['Iter-12000-acc-0.3875.pth','Iter-8800-acc-0.3876.pth',
    #                 'Iter-11200-acc-0.3857.pth','Iter-10400-acc-0.3835.pth'] # data 1
    model_name = ['Iter-240-acc-0.018.pth']  ## 测试代码

    run_dev(model_name,debug_flag,data_version)
    data_predict = get_predict_data(debug_flag,data_version)
    # data_predict = get_val_data(debug_flag,data_version)   ## 验证集成效果
    for name in model_name:
        logger.info(name)
        run_predict_crf(name,data_predict,data_version)



