import json
import logging
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn
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
    return torch.stack(features)


def includes_nan(tensor):
    return (tensor != tensor).any()


class ActivityNet_TrainVal(Dataset):
    def __init__(self, root_dir, ann_dir, feature_dirs, tokenizer, mode="train"):
        self.tokenizer = tokenizer
        self.feature_dirs = [os.path.join(root_dir, ftdir) for ftdir in feature_dirs]

        # load annotation files
        assert mode in ("train", "val_1", "val_2")
        with open(os.path.join(root_dir, ann_dir, f"{mode}_fps.json")) as f:
            ann = json.load(f)

        self.data = []
        failcnt = 0
        for id, obj in ann.items():
            ft_paths = [
                os.path.join(ftdir, "{}.pth".format(id)) for ftdir in self.feature_dirs
            ]
            n_actions = len(obj["sentences"])
            if not all([os.path.exists(path) for path in ft_paths]):
                # print(f'{id} does not exist in dataset features')
                failcnt += 1
                continue
            if "fps" not in obj.keys():
                failcnt += 1
                # print(f'{id} does not have framerate data')
                continue
            meta = {}
            meta["id"] = id
            ft_paths = [
                os.path.join(ft_path, "{}.pth".format(id))
                for ft_path in self.feature_dirs
            ]
            # list( feature_dir/v_xxxxxxxxxx.pth )
            meta["feature_paths"] = ft_paths
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
            }
        """
        feature_paths = self.data[index]["feature_paths"]
        n_actions = self.data[index]["n_actions"]
        raw_captions = self.data[index]["sentences"]
        timestamps = self.data[index]["timestamps"]
        # torch.Tensor (n_actions, max_seqlen)
        captions = self.tokenizer.encode(raw_captions)
        fps = self.data[index]["fps"]
        id = self.data[index]["id"]

        features = []
        for feature_path in feature_paths:
            # for 0.5 fps features
            if "resnet" in feature_path or "flow" in feature_path:
                # get features for frame
                # torch.Tensor (n_actions, C')
                feature = get_video_features(feature_path, timestamps, 0.5)
                # workaround for having not enough expected frames
                if includes_nan(feature):
                    print("skipping id {}: timestamp is {}".format(id, timestamps))
                    return self.__getitem__(index + 1)
            else:
                # get features for video
                # torch.Tensor (n_actions, C)
                feature = get_video_features(feature_path, timestamps, fps)
                # workaround for having not enough expected frames
                if includes_nan(feature):
                    print("skipping id {}: timestamp is {}".format(id, timestamps))
                    return self.__getitem__(index + 1)
            features.append(feature)
        # concatenate features along channel dimension
        features = torch.cat(features, dim=-1)

        return {
            "id": id,
            "n_actions": n_actions,
            "feature": features,
            "raw_caption": raw_captions,
            "caption": captions,
            "timestamp": timestamps,
        }

    def __len__(self):
        return len(self.data)


def collate_trainval(datalist):
    ids = []
    n_actions = []
    feature = []
    raw_captions = []
    captions = []
    timestamps = []
    for data in datalist:
        ids.append(data["id"])
        n_actions.append(data["n_actions"])
        feature.append(data["feature"])
        raw_captions.extend(data["raw_caption"])
        captions.append(data["caption"])
        timestamps.extend(data["timestamp"])
    n_actions = torch.tensor(n_actions, dtype=torch.long)
    # (sum(n_actions), C)
    feature = torch.cat(feature, dim=0)
    # (sum(n_actions), max_seqlen)
    captions = torch.cat(captions, dim=0)

    return {
        "id": ids,
        "n_actions": n_actions,
        "feature": feature,
        "raw_caption": raw_captions,
        "caption": captions,
        "timestamp": timestamps,
    }


if __name__ == "__main__":
    root_path = "/groups1/gaa50131/datasets/ActivityNet"
    ann_dir = "captions"
    feature_dir = "features/r50_k700"
    capfile = os.path.join(root_path, ann_dir, "captions.txt")
    tokenizer = BasicTokenizer(min_freq=3, max_len=30)
    tokenizer.from_textfile(capfile)
    ds = ActivityNet_TrainVal(root_path, ann_dir, feature_dir, tokenizer, "train")
    print(len(ds))
    for i in range(10):
        print(ds[i]["feature"].size())
    loader = DataLoader(ds, batch_size=5, collate_fn=collate_trainval)
    for data in loader:
        print(data)
        break
