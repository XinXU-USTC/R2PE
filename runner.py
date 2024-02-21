import os
import argparse
import random
import sys
import io

import numpy as np

import torch
import time
import yaml
import logging
import json
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve
from tqdm import tqdm


CLS_DATASET = ['play', 'StrategyQA', 'Fever', 'physics']


def find_most_frequent(lst):
    counter = Counter(lst)
    most_common = counter.most_common(1)
    return most_common[0][1] if most_common else 0

def save2jsonl(name, data):
    with open(name, 'w') as file:
        for dict_obj in data:
            json_str = json.dumps(dict_obj)
            file.write(json_str + '\n')

def readjsonl2list(name):
    data = []  # Create an empty list to store the dictionaries

    with open(name, 'r') as file:
        for line in file:
            dict_obj = json.loads(line)
            data.append(dict_obj)
    return data

def main():

    parser = argparse.ArgumentParser()
    # GPU
    parser.add_argument('--gpu', type=int, default=0, help="using gpu id")
    # Dataset
    parser.add_argument('--dataset', type=str, default='HotpotQA', help="Dataset name, gsm8k, math, StrategyQA, play, physics, Fever, 2WikiMultihop or HotpotQA.")
    # Model
    parser.add_argument('--model', type=str, default='gpt-4-0314', help="LLM name, e.g., text-davinci-003.")
    # Method
    parser.add_argument('--method', type=str, default='ADS', help="Method name for R2PE, ADS or PDS.")
    args = parser.parse_args()

    # set logger
    logger = logging.getLogger(name='R2PE')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(name)s] >> %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # set seed 
    seed = 123
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    logger.info("Seed set!")

    # load dataset
    fp = os.path.join('data', args.dataset, args.model, 'test.jsonl')
    data = readjsonl2list(fp)
    logger.info('Load {} examples from {}'.format(len(data), fp))

    # load nli model
    if args.method in ['PDS']:
        from nli import NLI
        # Device setting
        device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
        device = torch.device(device_str)  
        sent_nli = NLI(device=device, granularity='sentence', nli_model='mnli')
        logger.info(f"Device set to {device_str}.")

    # predict
    con = []
    gt = []
    for ex in tqdm(data, desc='Predicting'):
        answers, responses, label = ex['answers'], ex['responses'], ex['label']
        #print(responses, answers)
        gt.append(not label)
        if args.method == 'ADS':
            threshold = 4.5 if args.dataset in CLS_DATASET else 2.5
            ads = find_most_frequent(answers)
            #print(ex['id'], ads)
            con.append(ads < threshold)
        elif args.method == 'PDS':
            # compute ENS
            scores, mean_score = sent_nli.score(responses)
            ads = find_most_frequent(answers)
            pds_aux = np.min(mean_score)
            pds = ((ads-2.5)/2.5 + pds_aux) / 2
            #print(ens, pds, ads, ex['id'])
            if args.dataset in CLS_DATASET:
                threshold = 0.4
            else:
                threshold = 0.0
            if args.dataset == 'physics' and args.model == 'text-davinci-003':
                threshold = 0.25
            if args.dataset == 'Fever' and args.model == 'gemini-pro':
                threshold = 0.15  
            con.append(pds < threshold)
        else:
            raise NotImplementedError






    logger.info('PRECISION: {:.2f}\t'.format(100*precision_score(gt,  con))+\
                                 'RECALL: {:.2f}\t'.format(100*recall_score(gt, con))+\
                                 'F1: {:.2f}\t'.format(100*f1_score(gt, con)))
    


if __name__ == '__main__':
    main()

