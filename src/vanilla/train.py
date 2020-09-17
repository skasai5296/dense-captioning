import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from addict import Dict
from nlgeval import NLGEval
from torch.utils.data import DataLoader

import wandb
from dataset import ActivityNet_TrainVal, collate_trainval
from model import AttentionDecoder, SimpleDecoder
from utils import ModelSaver, Timer
from vocab import BasicTokenizer


def train_epoch(loader, model, optimizer, loss_weights, device, ep):
    epoch_timer = Timer()
    model.train()
    for it, data in enumerate(loader):
        feature = data["feature"]
        caption = data["caption"]
        length = data["n_actions"]
        feature = feature.to(device)
        caption = caption.to(device)
        length = length.to(device)

        optimizer.zero_grad()
        losses = model(feature, caption, length)
        lossstr = ""
        cumloss = 0
        for loss_name, loss in losses.items():
            # for gathering in dataparallel
            loss_val = loss.mean()
            if loss_name in loss_weights:
                weight = loss_weights[loss_name]
            else:
                weight = 1.0
            if CONFIG.use_wandb:
                wandb.log({loss_name: loss_val.item()})
            lossstr += " {}: {:.6f} |".format(loss_name, loss_val.item())
            cumloss += loss_val * weight
        cumloss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        if it % 10 == 9:
            print(
                "epoch {} | iter {} / {} |{}".format(
                    epoch_timer, it + 1, len(loader), lossstr
                )
            )


def validate(loader, model, tokenizer, evaluator, device, max_samples):
    gt_list = []
    ans_list = []
    val_timer = Timer()
    val_iters = (max_samples + loader.batch_size - 1) // loader.batch_size
    model.eval()
    for it, data in enumerate(loader):
        feature = data["feature"]
        raw_caption = data["raw_caption"]
        feature = feature.to(device)

        with torch.no_grad():
            if hasattr(model, "module"):
                decoded = model.module.sample(feature)
            else:
                decoded = model.sample(feature)
            decoded = torch.argmax(decoded, dim=1)
        generated = tokenizer.decode(decoded)

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

    CONFIG.hyperparam.misc.gpu_ids = list(map(str, CONFIG.hyperparam.misc.gpu_ids))
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(CONFIG.hyperparam.misc.gpu_ids)

    if CONFIG.hyperparam.misc.random_seed != 0:
        random.seed(CONFIG.hyperparam.misc.random_seed)
        np.random.seed(CONFIG.hyperparam.misc.random_seed)
        torch.manual_seed(CONFIG.hyperparam.misc.random_seed)
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
        min_freq=CONFIG.hyperparam.tokenizer.min_freq,
        max_len=CONFIG.hyperparam.tokenizer.max_len,
    )
    tokenizer.from_textfile(
        os.path.join(CONFIG.path.root, CONFIG.path.annotation, "captions.txt")
    )
    train_dataset = ActivityNet_TrainVal(
        CONFIG.path.root,
        CONFIG.path.annotation,
        CONFIG.path.feature,
        tokenizer,
        "train",
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG.hyperparam.misc.batch_size,
        shuffle=True,
        collate_fn=collate_trainval,
    )
    val_dataset = ActivityNet_TrainVal(
        CONFIG.path.root,
        CONFIG.path.annotation,
        CONFIG.path.feature,
        tokenizer,
        "val_1",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG.hyperparam.misc.batch_size,
        shuffle=True,
        collate_fn=collate_trainval,
    )
    print("done!")
    #########################################################

    ############### model, optimizer ########################
    print("loading model and optimizer...")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("using GPU numbers {}".format(CONFIG.hyperparam.misc.gpu_ids))
    else:
        device = torch.device("cpu")
        print("using CPU")
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
        if offset_ep > CONFIG.hyperparam.misc.max_epoch:
            raise RuntimeError(
                "trying to restart at epoch {} while max training is set to {} \
                epochs".format(
                    offset_ep, CONFIG.hyperparam.misc.max_epoch
                )
            )
    print("done!")
    ########################################################

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    if CONFIG.use_wandb:
        wandb.watch(model)

    ################### training loop #####################
    for ep in range(offset_ep - 1, CONFIG.hyperparam.misc.max_epoch):
        print("global {} | begin training for epoch {}".format(global_timer, ep + 1))
        train_epoch(train_loader, model, optimizer, CONFIG.loss_weight, device, ep)
        print(
            "global {} | done with training for epoch {}, beginning validation".format(
                global_timer, ep + 1
            )
        )
        metrics = validate(
            val_loader,
            model,
            tokenizer,
            evaluator,
            device,
            CONFIG.hyperparam.misc.val_samples,
        )
        if "METEOR" in metrics.keys():
            saver.save_ckpt_if_best(model, optimizer, metrics["METEOR"])
        print("global {} | end epoch {}".format(global_timer, ep + 1))
    print("done training!")
    #######################################################
