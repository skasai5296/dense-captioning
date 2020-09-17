import json
import logging
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from addict import Dict
from torch.nn.utils.rnn import pack_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

from vocab import BasicTokenizer


def get_video_features(path, timestamps, fps):
    # (C, T)
    feature = torch.load(path)
    assert feature.dim() == 2
    assert feature.size(0) in [256, 1024, 2048]
    assert not includes_nan(feature)

    framestamps = []
    for t in timestamps:
        startframe = max(0, int(t[0] * fps))
        endframe = int(t[1] * fps)
        framestamps.append([startframe, endframe])
    # cut features with framestamps and mean pool
    features = []
    for fs in framestamps:
        mean_f = feature[:, fs[0] : fs[1] + 1].mean(1)
        features.append(mean_f)
    # (n_actions, C)
    features = torch.stack(features)
    return features


def pad_feature(feature, maxlen):
    # (n_actions, D)
    assert feature.dim() == 2
    T, D = feature.size()
    out = torch.zeros(maxlen, D)
    out[: min(maxlen, T), :] = feature[: min(maxlen, T), :]
    return out


def pad_caption(caption, maxlen, pad_idx):
    # (n_actions, S)
    assert caption.dim() == 2
    T, D = caption.size()
    out = torch.full((maxlen, D), pad_idx, dtype=torch.long)
    out[: min(maxlen, T), :] = caption[: min(maxlen, T), :]
    return out


def includes_nan(tensor):
    return (tensor != tensor).any()


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
            # not good frame data
            if id == "v_rhOtqArO-3Y":
                continue
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
            meta["fps"] = (
                1.5
                if "resnet" in self.feature_dir or "flow" in self.feature_dir
                else obj["fps"]
            )
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
                'n_actions':        int;                            number of actions in video
                'feature' :         torch.Tensor(max_actions, C);   features for each action
                'raw_caption':      list(str);                      list of captions
                'caption':          torch.Tensor(max_actions, S);     caption ids
                'length':           torch.Tensor(max_actions);        length of captions
                'timestamp':        list([int, int]);               shows the beginning and end frames of action
            }
        """
        n_actions = min(self.data[index]["n_actions"], self.CONFIG.dim.max_actions)
        raw_captions = self.data[index]["sentences"]
        timestamps = self.data[index]["timestamps"]
        feature_path = self.data[index]["feature_path"]
        # torch.Tensor (n_actions, S)
        captions = self.tokenizer.encode(raw_captions)
        # torch.Tensor (max_actions, S)
        captions = pad_caption(
            captions, self.CONFIG.dim.max_actions, self.tokenizer.pad_idx
        )
        length = captions.ne(self.tokenizer.pad_idx).sum(dim=1)
        fps = self.data[index]["fps"]
        id = self.data[index]["id"]

        # torch.Tensor (n_actions, C)
        feature = get_video_features(feature_path, timestamps, fps)
        assert (feature == feature).all(), f"nan in features from id {id}"
        # torch.Tensor (max_actions, C)
        feature = pad_feature(feature, self.CONFIG.dim.max_actions)

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


def collater(datalist):
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
    # (bs)
    n_actions = torch.tensor(n_actions, dtype=torch.long)
    # (bs, max_actions, C)
    feature = torch.stack(feature)
    # (bs, max_actions)
    length = torch.stack(length)
    # (bs, max_actions, S)
    caption = torch.stack(caption)

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
    # for i in range(10):
    #    print(ds[i]["feature"].size())
    loader = DataLoader(ds, batch_size=5, collate_fn=collater)
    for data in loader:
        print(data)
        print(data["feature"].size())
        print(data["feature"].sum())
        print(includes_nan(data["feature"]))
        break
