import argparse
import copy
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import yaml
from addict import Dict
from nlgeval import NLGEval
from torch.utils.data import DataLoader

from dataset import ActivityNet_TrainVal, collater
from model import Captioning
from utils import ModelSaver, Timer
from vocab import BasicTokenizer


def train_epoch(loader, model, criterion, optimizer, device, CONFIG, ep):
    epoch_timer = Timer()
    model.train()
    for it, data in enumerate(loader):
        feature = data["feature"]
        caption = data["caption"]
        n_actions = data["n_actions"]
        length = data["length"]
        feature = feature.to(device)
        caption = caption.to(device)
        n_actions = n_actions.to(device)
        length = length.to(device)

        optimizer.zero_grad()
        logits = model(feature, n_actions, caption, length)
        loss = criterion(logits, caption)
        if CONFIG.use_wandb:
            wandb.log({"XELoss": loss.item()})
        lossstr = " XELoss: {:.6f} |".format(loss.item())
        if loss != loss:
            sys.exit(0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        if it % 10 == 9:
            print(
                "epoch {} | iter {} / {} |{}".format(
                    epoch_timer, it + 1, len(loader), lossstr
                )
            )


def validate(loader, model, tokenizer, evaluator, device, CONFIG):
    gt_list = []
    ans_list = []
    val_timer = Timer()
    max_samples = CONFIG.misc.val_samples
    val_iters = (max_samples + loader.batch_size - 1) // loader.batch_size
    model.eval()
    for it, data in enumerate(loader):
        feature = data["feature"]
        n_actions = data["n_actions"]
        raw_caption = data["raw_caption"]
        feature = feature.to(device)
        n_actions = n_actions.to(device)

        with torch.no_grad():
            decoded = model(feature, n_actions)
            decoded = torch.argmax(decoded, dim=1)
        B, A, C = decoded.size()
        generated = tokenizer.decode(decoded.view(B, A * C))

        gt_list.extend(raw_caption)
        ans_list.extend(generated)
        if it % 10 == 9:
            print("validation {} | iter {} / {}".format(val_timer, it + 1, val_iters))
            gts = raw_caption[:5]
            hyps = generated[:5]
            for gt, hyp in zip(gts, hyps):
                print("ground truth: {}, sampled: {}".format(gt, hyp))
        # only iterate for enough samples
        if it == val_iters - 1:
            break
    metrics = {}
    print("---METRICS---")
    gt_list = [gt_list[:max_samples]]
    ans_list = ans_list[:max_samples]
    metrics = evaluator.compute_metrics(ref_list=gt_list, hyp_list=ans_list)
    for k, v in metrics.items():
        print("{}:\t\t{}".format(k, v))
    print("---METRICS---")
    if CONFIG.use_wandb:
        wandb.log(metrics)
    return metrics


if __name__ == "__main__":
    global_timer = Timer()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="path to configuration yml \
        file",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="denotes if to continue training, will use config",
    )
    opt = parser.parse_args()
    CONFIG = Dict(yaml.safe_load(open(opt.config)))

    CONFIG.hyp.gpu_ids = list(map(str, CONFIG.hyp.gpu_ids))
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(CONFIG.hyp.gpu_ids)

    if CONFIG.misc.random_seed != 0:
        random.seed(CONFIG.misc.random_seed)
        np.random.seed(CONFIG.misc.random_seed)
        torch.manual_seed(CONFIG.misc.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if CONFIG.use_wandb:
        wandb.init(config=CONFIG, project="anet2")

    #################### prepare logs #######################
    outdir = os.path.join(CONFIG.path.output, CONFIG.name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    #########################################################

    ################# tokenizer and dataset #################
    print("loading tokenizer and dataset...")
    tokenizer = BasicTokenizer(
        min_freq=CONFIG.hyp.min_freq, max_len=CONFIG.dim.max_caption_len,
    )
    tokenizer.from_textfile(
        os.path.join(CONFIG.path.root, CONFIG.path.annotation, "captions.txt")
    )
    train_dataset = ActivityNet_TrainVal(CONFIG, tokenizer, "train",)
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG.dim.B,
        shuffle=True,
        collate_fn=collater,
        num_workers=CONFIG.misc.num_workers,
    )
    val_dataset = ActivityNet_TrainVal(CONFIG, tokenizer, "val_1",)
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG.dim.B,
        shuffle=True,
        collate_fn=collater,
        num_workers=CONFIG.misc.num_workers,
    )
    print("done!")
    #########################################################

    ############### model, optimizer ########################
    print("loading model and optimizer...")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("using GPU numbers {}".format(CONFIG.hyp.gpu_ids))
    else:
        device = torch.device("cpu")
        print("using CPU")
    model = Captioning(
        CONFIG, V=len(tokenizer), bos_idx=tokenizer.bos_idx, eos_idx=tokenizer.eos_idx
    )
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx)
    # criterion = CrossEntropy_OnlyAction()
    optimizer = getattr(optim, CONFIG.optim.name.capitalize())(
        model.parameters(),
        lr=CONFIG.optim.lr,
        betas=(CONFIG.optim.beta1, CONFIG.optim.beta2,),
        weight_decay=CONFIG.optim.weight_decay,
    )
    print("done!")
    #########################################################

    ################# evaluator, saver ######################
    print("loading evaluator and model saver...")
    evaluator = NLGEval()
    # evaluator = NLGEval(metrics_to_omit=["METEOR"])
    model_path = os.path.join(outdir, "best_score.ckpt")
    saver = ModelSaver(model_path, init_val=0)
    offset_ep = 1
    if opt.resume:
        offset_ep = saver.load_ckpt(model, optimizer, device)
        if offset_ep > CONFIG.misc.max_epoch:
            raise RuntimeError(
                "trying to restart at epoch {} while max training is set to {} \
                epochs".format(
                    offset_ep, CONFIG.misc.max_epoch
                )
            )
    print("done!")
    ########################################################

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    if CONFIG.use_wandb:
        wandb.watch(model)

    ################### training loop #####################
    for ep in range(offset_ep - 1, CONFIG.misc.max_epoch):
        print("global {} | begin training for epoch {}".format(global_timer, ep + 1))
        train_epoch(train_loader, model, criterion, optimizer, device, CONFIG, ep)
        print(
            "global {} | done with training for epoch {}, beginning validation".format(
                global_timer, ep + 1
            )
        )
        metrics = validate(val_loader, model, tokenizer, evaluator, device, CONFIG)
        if "METEOR" in metrics.keys():
            saver.save_ckpt_if_best(model, optimizer, metrics["METEOR"])
        print("global {} | end epoch {}".format(global_timer, ep + 1))
    print("done training!")
    #######################################################
