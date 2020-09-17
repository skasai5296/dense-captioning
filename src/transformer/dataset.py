import json
import logging
import os
import pickle
import sys
import time

import numpy as np
import yaml

import torch
import torch.nn as nn
from addict import Dict
from torch.utils.data import DataLoader, Dataset
from vocab import BasicTokenizer


def get_video_features(path):
    # (C, T)
    feature = torch.load(path)
    assert feature.dim() == 2
    assert feature.size(0) in [256, 1024, 2048]
    assert not includes_nan(feature)
    return feature


def includes_nan(tensor):
    return (tensor != tensor).any()


def pad_feature(feature, maxlen):
    padlen = maxlen - feature.size(0)
    if padlen <= 0:
        return feature[:maxlen]
    else:
        return torch.cat([feature, torch.zeros(padlen, feature.size(1))], dim=0)


class ActivityNet_TrainVal(Dataset):
    def __init__(self, CONFIG, tokenizer, mode="train"):
        self.tokenizer = tokenizer
        self.CONFIG = CONFIG
        self.feature_dir = os.path.join(CONFIG.path.root, CONFIG.path.feature)

        # load annotation files
        assert mode in ("train", "val_1", "val_2")
        with open(
            os.path.join(CONFIG.path.root, CONFIG.path.annotation, f"{mode}_fps.json")
        ) as f:
            ann = json.load(f)

        self.data = []
        failcnt = 0
        for id, obj in ann.items():
            ft_path = os.path.join(self.feature_dir, "{}.pth".format(id))
            n_actions = len(obj["sentences"])
            if "fps" not in obj.keys():
                failcnt += 1
                # print(f'{id} does not have framerate data')
                continue
            if not os.path.exists(ft_path):
                failcnt += 1
                # print(f'{id} does not have features')
                continue
            meta = {}
            meta["id"] = id
            os.path.join(ft_path, "{}.pth".format(id))
            # self.feature_dir/v_xxxxxxxxxx.pth
            meta["feature_path"] = ft_path
            meta["sentences"] = [sentence.strip() for sentence in obj["sentences"]]
            meta["timestamps"] = obj["timestamps"]
            meta["fps"] = obj["fps"]
            meta["n_actions"] = n_actions
            self.data.append(meta)
        print("failed to load {}/{} for {} dataset".format(failcnt, len(ann), mode))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            {
                'id':               str;
                'n_actions':        int;                        number of actions in video
                'feature' :         torch.Tensor(n_actions, C); features for each action
                'raw_caption':      list(str);                  list of captions
                'caption':          torch.Tensor(n_actions, S); caption ids
                'timestamp':        list([int, int]);           shows the beginning and end frames of action
                'length':           torch.Tensor(n_actions);    number of features
            }
        """
        n_actions = self.data[index]["n_actions"]
        raw_captions = self.data[index]["sentences"]
        # for joining sentences
        raw_captions = [" ".join(raw_captions)]
        timestamps = self.data[index]["timestamps"]
        feature_path = self.data[index]["feature_path"]
        # torch.Tensor (1, T)
        captions = self.tokenizer.encode(raw_captions)
        fps = self.data[index]["fps"]
        id = self.data[index]["id"]

        # torch.Tensor (S, C)
        feature = get_video_features(feature_path).t()
        length = min(feature.size(0), self.CONFIG.hyperparam.transformer.input_seqlen)
        feature = pad_feature(feature, self.CONFIG.hyperparam.transformer.input_seqlen)

        return {
            "id": id,
            "n_actions": n_actions,
            "feature": feature,
            "raw_caption": raw_captions,
            "caption": captions,
            "timestamp": timestamps,
            "fps": fps,
            "length": length,
        }

    def __len__(self):
        return len(self.data)


def collate_trainval(datalist):
    ids = []
    n_actions = []
    feature = []
    raw_caption = []
    caption = []
    timestamps = []
    fps = []
    length = []
    for data in datalist:
        ids.append(data["id"])
        n_actions.append(data["n_actions"])
        feature.append(data["feature"])
        raw_caption.extend(data["raw_caption"])
        caption.append(data["caption"])
        timestamps.extend(data["timestamp"])
        fps.append(data["fps"])
        length.append(data["length"])
    n_actions = torch.tensor(n_actions, dtype=torch.long)
    # (bs, S, C)
    feature = torch.stack(feature, dim=0)
    # (bs)
    length = torch.tensor(length, dtype=torch.long)
    # (bs, T)
    caption = torch.cat(caption, dim=0)

    return {
        "id": ids,
        "n_actions": n_actions,
        "feature": feature,
        "raw_caption": raw_caption,
        "caption": caption,
        "timestamp": timestamps,
        "fps": fps,
        "length": length,
    }


if __name__ == "__main__":
    cfg_path = "../../cfg/debug_aist.yml"
    CONFIG = Dict(yaml.safe_load(open(cfg_path)))
    capfile = os.path.join(CONFIG.path.root, CONFIG.path.annotation, "captions.txt")
    tokenizer = BasicTokenizer(min_freq=3, max_len=30)
    tokenizer.from_textfile(capfile)
    ds = ActivityNet_TrainVal(CONFIG, tokenizer, "train")
    print(len(ds))
    for i in range(10):
        print(ds[i]["feature"].size())
    loader = DataLoader(ds, batch_size=5, collate_fn=collate_trainval)
    for data in loader:
        print(data)
        print(data["feature"].size())
        break
