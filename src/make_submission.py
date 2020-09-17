import argparse
import json
import logging
import os
import shutil
import sys
import time

import numpy as np
import torch
import yaml
from addict import Dict
from torch import nn, optim
from torch.utils.data import DataLoader

from dataset import ActivityNet_TrainVal, collate_trainval
from model import AttentionDecoder, SimpleDecoder
from utils import ModelSaver, Timer
from vocab import BasicTokenizer

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="path to configuration yml file"
    )
    parser.add_argument(
        "--loglevel",
        type=str,
        default="debug",
        help="denotes log level, should be one of [debug|info|warning|error|critical]",
    )
    opt = parser.parse_args()
    CONFIG = Dict(yaml.safe_load(open(opt.config)))

    #################### prepare logs #######################
    numeric_level = getattr(logging, opt.loglevel.upper())
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: {}".format(numeric_level))
    outdir = os.path.join(CONFIG.path.output, CONFIG.name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    logging.basicConfig(
        filename=os.path.join(outdir, "{}.log".format(opt.loglevel.lower())),
        level=numeric_level,
    )
    #########################################################

    ################# tokenizer and dataset #################
    logging.info("loading tokenizer and dataset...")
    tokenizer = BasicTokenizer(
        min_freq=CONFIG.hyperparam.tokenizer.min_freq,
        max_len=CONFIG.hyperparam.tokenizer.max_len,
    )
    tokenizer.from_textfile(
        os.path.join(CONFIG.path.root, CONFIG.path.annotation, "captions.txt")
    )
    dataset = ActivityNet_TrainVal(
        CONFIG.path.root,
        CONFIG.path.annotation,
        CONFIG.path.feature,
        tokenizer,
        "val_1",
    )
    loader = DataLoader(
        dataset,
        batch_size=CONFIG.hyperparam.misc.batch_size,
        shuffle=False,
        collate_fn=collate_trainval,
    )
    logging.info("done!")
    #########################################################

    ############### model, optimizer ########################
    logging.info("loading model and optimizer...")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info("using {} GPU(s)".format(torch.cuda.device_count()))
    else:
        device = torch.device("cpu")
        logging.info("using CPU")
    if CONFIG.hyperparam.attention:
        model = AttentionDecoder(
            feature_dim=CONFIG.hyperparam.feature.dim,
            emb_dim=CONFIG.hyperparam.rnn.word_emb_dim,
            memory_dim=CONFIG.hyperparam.rnn.memory_dim,
            vocab_size=len(tokenizer),
            max_seqlen=CONFIG.hyperparam.tokenizer.max_len,
            dropout_p=CONFIG.hyperparam.rnn.dropout_prob,
            ss_prob=CONFIG.hyperparam.rnn.scheduled_sampling_prob,
            bos_idx=tokenizer.bosidx,
            pad_idx=tokenizer.padidx,
        )
    else:
        model = SimpleDecoder(
            feature_dim=CONFIG.hyperparam.feature.dim,
            emb_dim=CONFIG.hyperparam.rnn.word_emb_dim,
            memory_dim=CONFIG.hyperparam.rnn.memory_dim,
            vocab_size=len(tokenizer),
            max_seqlen=CONFIG.hyperparam.tokenizer.max_len,
            dropout_p=CONFIG.hyperparam.rnn.dropout_prob,
            ss_prob=CONFIG.hyperparam.rnn.scheduled_sampling_prob,
            bos_idx=tokenizer.bosidx,
            pad_idx=tokenizer.padidx,
        )
    model = model.to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=CONFIG.hyperparam.optimization.lr,
        betas=(
            CONFIG.hyperparam.optimization.beta1,
            CONFIG.hyperparam.optimization.beta2,
        ),
        weight_decay=CONFIG.hyperparam.optimization.weight_decay,
    )
    logging.info("done!")
    #########################################################

    ################# load model params ######################
    logging.info("loading model params...")
    model_path = os.path.join(outdir, "best_score.ckpt")
    saver = ModelSaver(model_path)
    offset_ep = saver.load_ckpt(model, optimizer, device)
    if offset_ep == 1:
        raise RuntimeError("aborting, no pretrained model")
    logging.info("done!")
    ##########################################################

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    ################# make submission file ######################
    logging.info("making submission file...")
    submission = {
        "version": "VERSION 1.3",
        "external_data": {
            "used": True,
            "details": "Excluding the last fc layer, the video encoding model (3D-ResneXt-101) is pre-trained on the Kinetics-400 training set",
        },
    }
    obj = {}
    for it, data in enumerate(loader):
        ids = data["id"]
        n_actions = data["n_actions"]
        timestamps = data["timestamp"]
        feature = data["feature"]
        feature = feature.to(device)

        with torch.no_grad():
            if hasattr(model, "module"):
                decoded = model.module.sample(feature)
            else:
                decoded = model.sample(feature)
            decoded = torch.argmax(decoded, dim=1)
        generated = tokenizer.decode(decoded)
        done = 0
        for i, id in enumerate(ids):
            n_action = n_actions[i]
            caption = generated[done : done + n_action]
            timestamp = timestamps[done : done + n_action]
            res = [
                {"sentence": sen, "timestamp": ts}
                for sen, ts in zip(caption, timestamp)
            ]
            obj[id] = res
        if it % 1000 == 999:
            logging.info("{} / {}".format(it + 1, len(loader)))
    submission["results"] = obj
    ##############################################################

    submission_path = os.path.join(
        CONFIG.path.output, CONFIG.name, CONFIG.submission_file
    )
    with open(submission_path, "w") as f:
        json.dump(submission, f)
    logging.info("end creation of json file, saved at {}".format(submission_path))
