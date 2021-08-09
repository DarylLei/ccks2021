import codecs
import json
import os,sys
import numpy as np
import torch
sys.path.append('.')
from logger import logger
from sentence_transformers import SentenceTransformer, util
# model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
model = SentenceTransformer('paraphrase-mpnet-base-v2')


config_labels = ['ENT']
# config_case_dir = 'ccks/output/case/bad_case.txt'

def get_entities(seq):
    """
    Gets entities from sequence.
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        get_entities(seq)
        [('PER', 0, 1), ('LOC', 3, 3)]
    """
    # for nested list
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]
    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ['O']):
        tag = chunk[0]
        type_ = chunk.split('-')[-1]

        if end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin_offset, i - 1))
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_offset = i
        prev_tag = tag
        prev_type = type_

    return chunks


def end_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk ended between the previous and current word.
    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.
    Returns:
        chunk_end: boolean.
    """
    chunk_end = False

    if prev_tag == 'S':
        chunk_end = True
    # pred_label中可能出现这种情形
    if prev_tag == 'B' and tag == 'B':
        chunk_end = True
    if prev_tag == 'B' and tag == 'S':
        chunk_end = True
    if prev_tag == 'B' and tag == 'O':
        chunk_end = True
    if prev_tag == 'I' and tag == 'B':
        chunk_end = True
    if prev_tag == 'I' and tag == 'S':
        chunk_end = True
    if prev_tag == 'I' and tag == 'O':
        chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk started between the previous and current word.
    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.
    Returns:
        chunk_start: boolean.
    """
    chunk_start = False

    if tag == 'B':
        chunk_start = True
    if tag == 'S':
        chunk_start = True

    if prev_tag == 'S' and tag == 'I':
        chunk_start = True
    if prev_tag == 'O' and tag == 'I':
        chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    return chunk_start


def f1_score(y_true, y_pred, mode='dev'):
    """Compute the F1 score.
    The F1 score can be interpreted as a weighted average of the precision and
    recall, where an F1 score reaches its best value at 1 and worst score at 0.
    The relative contribution of precision and recall to the F1 score are
    equal. The formula for the F1 score is::
        F1 = 2 * (precision * recall) / (precision + recall)
    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.
    Returns:
        score : float.
    Example:
        y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        f1_score(y_true, y_pred)
        0.50
    """
    true_entities = set(get_entities(y_true))
    pred_entities = set(get_entities(y_pred))
    # tmp_c = true_entities & pred_entities
    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)
    nb_true = len(true_entities)

    p = nb_correct / nb_pred if nb_pred > 0 else 0
    r = nb_correct / nb_true if nb_true > 0 else 0
    score = 2 * p * r / (p + r) if p + r > 0 else 0
    if mode == 'dev':
        return float(p),float(r),float(score)
    else:
        f_score = {}
        for label in config_labels:
            true_entities_label = set()
            pred_entities_label = set()
            for t in true_entities:
                if t[0] == label:
                    true_entities_label.add(t)
            for p in pred_entities:
                if p[0] == label:
                    pred_entities_label.add(p)
            nb_correct_label = len(true_entities_label & pred_entities_label)
            nb_pred_label = len(pred_entities_label)
            nb_true_label = len(true_entities_label)

            p_label = nb_correct_label / nb_pred_label if nb_pred_label > 0 else 0
            r_label = nb_correct_label / nb_true_label if nb_true_label > 0 else 0
            score_label = 2 * p_label * r_label / (p_label + r_label) if p_label + r_label > 0 else 0
            f_score[label] = float(score_label) 
        return f_score, score


def f1_score_ac(y_pred,data,data_version):
    '''实际计算的f1值'''
    y_true = []
    n_ids = []
    with open("data/sem_tag/{}/val_nids.txt".format(data_version), "r") as inf:
        id_lines = inf.readlines()
        for item in id_lines:
            _id = item.strip().split('\t')[0]
            n_ids.append(_id)
            _tags = json.loads(item.strip().split('\t')[1])
            _tags = [ _id+"_"+_item["@value"]for _item in _tags]
            y_true.extend(_tags)
    y_pre_name = []
    for idx, p, _s in zip(n_ids,y_pred, data):
        tmp_len = len(_s)
        model_pre = get_entity_name(p[:tmp_len],_s)
        model_pre = [ idx +"_"+ ''.join(_item[0]) for _item in model_pre]
        y_pre_name.extend(model_pre)

    true_entities = set(y_true)
    pred_entities = set(y_pre_name)
    # tmp_c = true_entities & pred_entities
    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)
    nb_true = len(true_entities)

    p = nb_correct / nb_pred if nb_pred > 0 else 0
    r = nb_correct / nb_true if nb_true > 0 else 0
    score = 2 * p * r / (p + r) if p + r > 0 else 0
    return float(p), float(r), float(score) 


def modify_with_rule(y_pred,data):
    ## 所有标签
    sem_tags_all = []
    with codecs.open("ccks/data/analyze/sem_tag_vs_nids.txt",'r') as fr:
        tmp_data = json.load(fr)
        for key,value in tmp_data.items():
            sem_tags_all.extend(value['n_entities'])
    logger.info('sem_tags_all:{}'.format(len(sem_tags_all)))
    sem_tags_all = list(set(sem_tags_all))
    logger.info('去重 sem_tags_all:{}'.format(len(sem_tags_all)))
            
    key_word_with_u = {}
    with open("ccks/data/analyze/sem_split_with_u.txt", "r") as inf:
        split_with_u = json.load(inf)
        for item in split_with_u:
            if item == '':
                continue
            fir,sec = item.strip().split('\u0001')
            if fir in key_word_with_u:
                key_word_with_u[fir].append(item)
            else:
                key_word_with_u[fir] = [item]
    y_true = []
    n_ids = []
    with open("ccks/data/sem_tag_pre/add_asr/val_nids.txt", "r") as inf:
        id_lines = inf.readlines()
        for item in id_lines:
            _id = item.strip().split('\t')[0]
            n_ids.append(_id)
            _tags = json.loads(item.strip().split('\t')[1])
            _tags = [ _id+"_"+_item["@value"]for _item in _tags]
            y_true.extend(_tags)
    y_pre_name = []
    for i, (idx, p, _s) in enumerate(zip(n_ids,y_pred, data)):
        if i%500==0:
            print(i)
        tmp_len = len(_s)
        model_pre = get_entity_name(p[:tmp_len],_s)
        model_pre = [''.join(_item[0]) for _item in model_pre]

        new_add_tag = []
        for _itme_value in model_pre:
            if _itme_value in key_word_with_u:
                tmp_new = get_add_sem_tag(key_word_with_u[_itme_value],''.join(_s))
                new_add_tag.extend(tmp_new)
        model_pre = model_pre + new_add_tag
        
        # new_remove_tag = []
        # for _tmp_item in model_pre:
        #     if _tmp_item not in sem_tags_all:
        #         continue
        #     else:
        #         new_remove_tag.append(_tmp_item)
        # model_pre = new_remove_tag

        model_pre = [ idx +"_"+ _item for _item in model_pre]
        y_pre_name.extend(model_pre)

    true_entities = set(y_true)
    pred_entities = set(y_pre_name)
    # tmp_c = true_entities & pred_entities
    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)
    nb_true = len(true_entities)

    p = nb_correct / nb_pred if nb_pred > 0 else 0
    r = nb_correct / nb_true if nb_true > 0 else 0
    score = 2 * p * r / (p + r) if p + r > 0 else 0
    return float(p), float(r), float(score) 


def get_add_sem_tag(tag_list,sentence): 
    embeddings2 = model.encode([sentence[:100]], convert_to_tensor=True)
    cosine_scores = []
    for item in tag_list:
        embeddings1 = model.encode([item], convert_to_tensor=True)
        _scores = util.pytorch_cos_sim(embeddings1, embeddings2)
        cosine_scores.append(_scores.item())
    index_s = cosine_scores.index(max(cosine_scores))
    return [tag_list[index_s]]

def bad_case(y_true, y_pred, data,config_case_dir):
    bad_case_num = 0
    if not os.path.exists(config_case_dir):
            os.makedirs(config_case_dir)
            logger.info('create logger path:{}'.format(config_case_dir))
    config_case_dir = config_case_dir + '/bad_case.txt'
    output = open(config_case_dir, 'w')
    for idx, (t, p, _s) in enumerate(zip(y_true, y_pred, data)):
        tmp_len = len(_s)
        t = t[:tmp_len] 
        p = p[:tmp_len] 
        if t== p:
            continue
        else:
            bad_case_num += 1
            golden = get_entity_name(t[:tmp_len],_s)
            model_pre = get_entity_name(p[:tmp_len],_s)
            output.write("bad case " + str(idx) + ": \n")
            output.write("sentence: " + ''.join(_s) + "\n")
            # output.write("golden label: " + str(t) + "\n")
            golden_ner = [''.join(_item[0]) for _item in golden]
            output.write("entity in text: " + str(golden_ner) + "\n")
            model_pre_ner = [''.join(_item[0]) for _item in model_pre]
            # output.write("model pred: " + str(p) + "\n")
            output.write("entity pred: " + str(model_pre_ner) + "\n")
    logger.info("Bad Cases num:{}".format(bad_case_num))
    logger.info("--------Bad Cases reserved !--------")
    


def get_entity_name(seq,tokens):
    ent_list = get_entities(seq)
    ent_name = []
    for item in ent_list:
        ent_name.append([tokens[item[1]:item[2]+1],item[1],item[2]])
    return ent_name

if __name__ == "__main__":
    y_t = [['O', 'O', 'O', 'B-address', 'I-address', 'I-address', 'O'], ['B-name', 'I-name', 'O']]
    y_p = [['O', 'O', 'B-address', 'I-address', 'I-address', 'I-address', 'O'], ['B-name', 'I-name', 'O']]
    sent = [['十', '一', '月', '中', '山', '路', '电'], ['周', '静', '说']]
    bad_case(y_t, y_p, sent)