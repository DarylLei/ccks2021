import json,sys,os
sys.path.append('.')
os.environ['CUDA_VISIBLE_DEVICES']='0' 
from models import ccks_tag_model
import numpy as np
import torch
from torch import cuda
from data_loader.data_load_npy import build_dataloader,get_predict_data,get_label_name_dict,get_val_data
from logger import logger
from torch.utils.tensorboard import SummaryWriter 
import time
import torch.nn.functional as F
from sklearn.metrics import classification_report
import torch.nn as nn
import codecs


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    logger.info("Total:{}, Trainable:{}".format(total_num,trainable_num))
    return {'Total': total_num, 'Trainable': trainable_num}


class config_tmp:
    def __init__(self,data_version) -> None:
        self.train_batch_size = 8
        self.valid_batch_size = 8
        self.num_epochs = 5
        self.device = 'cuda' if cuda.is_available() else 'cpu'
        self.model_name = 'text_bert'
        self.data_version = data_version
        self.npy_path_dir_train = '/data1/lvzhengwei/git_code/torch_dir/ccks_data/Data/CCKS_dataset/tsn_features_train/'
        self.npy_path_dir_test = '/data1/lvzhengwei/git_code/torch_dir/ccks_data/test_b/tsn_features_test_b/' 
    def create_path(self):
        self.save_path = 'ccks/output_'+ self.data_version +'/' + self.model_name + '/saved_model/'
        self.log_path = 'ccks/output_'+ self.data_version +'/' + self.model_name + '/log_board/'
        for path in [self.save_path,self.log_path]:
          if not os.path.exists(path):
            os.makedirs(path)
            logger.info('create logger path:{}'.format(path))

def validation(config, model, dev_iter,labels_tags):
    model.eval()
    loss_total = 0
    fin_targets=[]
    fin_outputs=[]
    device = config.device
    with torch.no_grad():
        for _, data in enumerate(dev_iter):
            text_feature = data['text_feature']
            ids, mask, token_type_ids = text_feature.values()
            ids, mask, token_type_ids = ids.to(device), mask.to(device), token_type_ids.to(device)
            npy_feature = data['npy_feature'].to(device)
            labels = data['labels'].to(device)   
            output = model(ids, mask, token_type_ids,npy_feature,labels)
            loss,logits = output['loss'],output['logits']
            loss_total += loss.mean()
            tmp_fin_targets = labels.cpu().detach().numpy()
            pred = F.softmax(logits,dim=1)
            tmp_fin_outputs=torch.argmax(pred,dim=1).cpu().numpy()
            fin_targets.extend(tmp_fin_targets)
            fin_outputs.extend(tmp_fin_outputs)
    fin_outputs,fin_targets = np.array(fin_outputs),np.array(fin_targets)
    metric_result_json = classification_report(fin_targets, fin_outputs, zero_division=0, output_dict=True)
    _tmp_ = [ i for i in range(len(labels_tags))]
    metric_result = classification_report(fin_targets, fin_outputs,labels=_tmp_, target_names=labels_tags, zero_division=0)
  
    # logger.info('\n' + metric_result)
    train_acc = metric_result_json['accuracy']
    model.train()
    return train_acc, loss_total / len(dev_iter)

def run_train(config,debug_flag,level,data_version):
    logger.info("---#### start training ####---")
    seed_random = 5124
    np.random.seed(seed_random)
    torch.manual_seed(seed_random)
    torch.cuda.manual_seed_all(seed_random)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样
    logger.info('随机种子：{}'.format(seed_random))

    if level == 'level1':
        class_num = 33
        best_dev_acc = 0.72
        with codecs.open('data/class_tag/{}/level1_label.txt'.format(data_version), "r", encoding="utf-8") as inf:
          labels_tags = [line.strip() for line in inf.readlines()]
    elif level == 'level2':
        class_num = 310
        best_dev_acc = 0.56
        with codecs.open('data/class_tag/{}/level2_label.txt'.format(data_version), "r", encoding="utf-8") as inf:
          labels_tags = [line.strip() for line in inf.readlines()]

    # model = ccks_tag_model.Model(class_num)
    # model = ccks_tag_model.BERTMulti(class_num)
    model = ccks_tag_model.BERTMultiRDrop(class_num)

    
    config.model_name = model.model_name
    config.create_path()
    logger.info("---#### model_name :{} ####---".format(config.model_name))

    if torch.cuda.device_count() > 1:
      logger.info("Let's use {} GPUs!".format(torch.cuda.device_count()))
      config.train_batch_size = config.train_batch_size * int(torch.cuda.device_count())
      config.valid_batch_size = config.valid_batch_size * int(torch.cuda.device_count())
      # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
      model = nn.DataParallel(model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # model.cuda()  

    get_parameter_number(model)
    # print(model.parameters)
    train_iter, dev_iter = build_dataloader(config,debug_flag,level,data_version)
    model.train()
    # optimizer = torch.optim.Adam([  {'params':model.module.bertmodel.parameters(),'lr':1e-5},
    #                                 {'params':model.module.l3.parameters()},
    #                                 {'params':model.module.l2.parameters()}], lr=1e-3)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    total_batch = 0  # 记录进行到多少batch
    dev_loss,train_acc,dev_acc = 0.0,0.0,0.0
    dev_acc_list = []
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    if debug_flag:
      best_dev_acc = 0.
      val_step = 80
    else:
      val_step = 1600
    for epoch in range(config.num_epochs):
        logger.info('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        # scheduler.step() # 学习率衰减
        for i, data in enumerate(train_iter):
            text_feature = data['text_feature']
            ids, mask, token_type_ids = text_feature.values()
            ids, mask, token_type_ids = ids.to(device), mask.to(device), token_type_ids.to(device)
            npy_feature = data['npy_feature'].to(device)
            labels = data['labels'].to(device)   
            output = model(ids, mask, token_type_ids,npy_feature,labels)
            loss = output['loss'].mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if total_batch % 20 == 0 and total_batch > 0:
                logger.info('Iter:{}, epoch:{}/{}  train_loss:{:.4}'
                            .format(total_batch, epoch+1, config.num_epochs, loss.item()))
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                # writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)

            if total_batch %val_step == 0 and total_batch > 0 :
                  dev_acc, dev_loss = validation(config, model, dev_iter,labels_tags)
                  logger.info('----#### eval ####----')
                  logger.info('Val loss: {:.4}  val_acc: {:.4} '.format(dev_loss,dev_acc))
                  dev_acc_list.append({"iter":total_batch,'dev_acc':dev_acc})
                  if dev_acc > best_dev_acc:
                      best_dev_acc = dev_acc
                      torch.save(model, config.save_path + level+'-Iter-{}-acc-{:.4}.pth'.format(total_batch,best_dev_acc))
                
            total_batch += 1
    writer.close()
    dev_acc_list = sorted(dev_acc_list,key=lambda x:x['dev_acc'],reverse=True)
    show_line = ''
    for item in dev_acc_list[:3]:
            show_line += '   iter:{}  dev_acc:{:.4}\n'.format(item['iter'],item['dev_acc'])
    logger.info('dev top3 result:\n{}'.format(show_line))

def run_predict(name,test_data,level,data_version):
  import codecs,json
  path_load = 'ccks/output_{}/bert_multi_rdrop/saved_model/'.format(data_version) + name
  # path_load = 'ccks/output_{}/bert_multi/saved_model/'.format(data_version) + name
  model = torch.load(path_load) 
  logger.info("Let's use {} GPUs!".format(torch.cuda.device_count()))
  ## 判断是单卡还是多卡模型
  if isinstance(model,torch.nn.DataParallel) and torch.cuda.device_count() == 1:
    model = model.module
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  logger.info('use device:{}'.format(device))
  model.to(device)

  if level == 'level1':
    label_2_id,id_2_label= get_label_name_dict('data/class_tag/{}/level1_label.txt'.format(data_version))
  elif level == 'level2':
    label_2_id,id_2_label= get_label_name_dict('data/class_tag/{}/level2_label.txt'.format(data_version))
  
  top1_results = {}
  for i, data in enumerate(test_data):
    if i%100 == 0 and i>0:
      logger.info(i)
    model.eval()
    with torch.no_grad():
        text_feature = data['text_feature']
        ids, mask, token_type_ids = text_feature.values()
        video_id = data['video_id']
        labels = data['labels']
        npy_feature = data['npy_feature'].to(device)
        output = model(ids.to(device), mask.to(device), token_type_ids.to(device),npy_feature)
        logits = output['logits']
        pred = F.softmax(logits,dim=1)
        # pred_index=torch.argmax(pred,dim=1).cpu().numpy()
        pred_probablity,pred_index = torch.max(pred,dim=1)
        for _index,_prob,_id in zip(pred_index,pred_probablity,video_id):
            top1_results[_id] = {
            "class_id": _index.item(),
            "class_name": id_2_label[_index.item()],
            "probability": _prob.item(),
            }
 
  for key,value in top1_results.items():
    if '_其他' in value['class_name']:
      top1_results[key]['class_name'] = '其他'
 
  # 保存结果
  path_save = 'ccks/output_{}/result/tag_cls/'.format(data_version) + level + '/'
  if not os.path.exists(path_save):
    os.makedirs(path_save)
    logger.info('create logger path:{}'.format(path_save))
  with codecs.open(path_save + name + '.json','w',"utf-8") as f:
      json.dump(top1_results, f, ensure_ascii=False, indent=4)
  logger.info('predict finished !')

def run_dev(config,name,debug_flag,level,data_version):
  logger.info('######---run_dev---######')
  logger.info(name)
  path_load = 'ccks/output_{}/bert_multi_rdrop/saved_model/'.format(data_version) + name[0]
  # path_load = 'ccks/output_{}/bert_multi/saved_model/'.format(data_version) + name[0]
  model = torch.load(path_load) 
  logger.info("Let's use {} GPUs!".format(torch.cuda.device_count()))
  ## 判断是单卡还是多卡模型
  if isinstance(model,torch.nn.DataParallel) and torch.cuda.device_count() == 1:
    model = model.module
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  logger.info('use device:{}'.format(device))
  model.to(device)
  if level == 'level1':
    label_2_id,id_2_label= get_label_name_dict('data/class_tag/{}/level1_label.txt'.format(data_version))
  elif level == 'level2':
    label_2_id,id_2_label= get_label_name_dict('data/class_tag/{}/level2_label.txt'.format(data_version))
  test_data = get_val_data(config,debug_flag,level,data_version)
  top1_results = {}
  fin_targets=[]
  fin_outputs=[]
  for i, data in enumerate(test_data):
    if i%100 == 0 and i>0:
      logger.info(i)
    model.eval()
    with torch.no_grad():
        text_feature = data['text_feature']
        ids, mask, token_type_ids = text_feature.values()
        video_id = data['video_id']
        labels = data['labels']
        npy_feature = data['npy_feature'].to(device)
        output = model(ids.to(device), mask.to(device), token_type_ids.to(device),npy_feature)
        logits = output['logits']
        pred = F.softmax(logits,dim=1)
        pred_probablity,pred_index = torch.max(pred,dim=1)

        tmp_fin_targets = labels.cpu().detach().numpy()
        tmp_fin_outputs=torch.argmax(pred,dim=1).cpu().numpy()
        fin_targets.extend(tmp_fin_targets)
        fin_outputs.extend(tmp_fin_outputs)

        for _index,_prob,_id,_true_label in zip(pred_index,pred_probablity,video_id,labels):
            top1_results[_id] = {
            "class_id": _index.item(),
            "class_name": id_2_label[_index.item()],
            "probability": _prob.item(),
            "true_label": id_2_label[_true_label.item()],
            }
  # 预测标签
  metric_result_json = classification_report(fin_targets, fin_outputs, zero_division=0, output_dict=True)
  _tmp_all = [[key,value]for key,value in id_2_label.items()]
  _tmp_,labels_tags = zip(*_tmp_all)
  metric_result = classification_report(fin_targets, fin_outputs,labels=_tmp_, target_names=labels_tags, zero_division=0)
  logger.info('\n' + metric_result)
  train_acc = metric_result_json['accuracy']
  logger.info('acc: {:.5}'.format(train_acc))
  

 





if __name__=='__main__':
    debug_flag = True
    level = 'level1'
    data_version = 'text_pre5'
    logger.info('分类级别:{}'.format(level))
    logger.info('data_version:{}'.format(data_version))
    config = config_tmp(data_version)
    ## 训练
    # run_train(config,debug_flag,level,data_version)
    
    if level == 'level1':
      # model_name = ['level1-Iter-11200-acc-0.7493.pth']  ## base data2
      # model_name = ['level1-Iter-16800-acc-0.7489.pth','level1-Iter-15200-acc-0.7457.pth'] ## data_add  data2
      # model_name = ['level1-Iter-12800-acc-0.7493.pth','level1-Iter-11200-acc-0.746.pth'] ## base  data3
      # model_name = ['level1-Iter-12800-acc-0.7526.pth','level1-Iter-15200-acc-0.7516.pth','level1-Iter-12800-acc-0.7496.pth'] ## base  data4
      # model_name = ['level1-Iter-17600-acc-0.7472.pth','level1-Iter-16000-acc-0.745.pth'] ## base  data4 2
      model_name = ['level1-Iter-160-acc-0.4351.pth'] ## 测试
    else:
      # model_name = ['level2-Iter-14400-acc-0.5753.pth','level2-Iter-12800-acc-0.5728.pth'] ## base data2
      # model_name = ['',''] ## data_add  data2
      # model_name = ['level2-Iter-14400-acc-0.5639.pth','level2-Iter-13600-acc-0.5627.pth'] ## base  data3
      # model_name = ['level2-Iter-19200-acc-0.579.pth','level2-Iter-16000-acc-0.5674.pth'] ## base  data4  2
      # model_name = ['level2-Iter-16000-acc-0.5798.pth','level2-Iter-14400-acc-0.5756.pth'] ## base  data4
      # model_name = ['level2-Iter-14400-acc-0.5714.pth'] ## base  data4  large
      model_name = ['level2-Iter-160-acc-0.2016.pth'] ## 测试

    ## 评估
    run_dev(config,model_name,debug_flag,level,data_version)

    ## 预测
    data_predict = get_predict_data(config,debug_flag,level,data_version)
    for name in model_name:
        logger.info(name)
        run_predict(name,data_predict,level,data_version)
    logger.info('finished all !')



