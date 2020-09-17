import json
import os
import pickle
import sys
import time

import spacy
import torch
import torchtext

spacy.load("en_core_web_md")


class BasicTokenizer:
    """
    Args:
        min_freq:   int;     word frequency threshold when building vocabulary
        max_len:    int;     max length when building tensors
    """

    def __init__(self, min_freq, max_len):
        self.min_freq = min_freq
        self.max_len = max_len
        self.field = torchtext.data.Field(
            sequential=True,
            init_token="<bos>",
            eos_token="<eos>",
            lower=True,
            fix_length=self.max_len,
            tokenize="spacy",
            batch_first=True,
        )

    """
    Build vocabulary from textfile.
    Sentences are separated by '\n'
    Args:
        textfile:   str;    path to textfile containing vocabulary
    """

    def from_textfile(self, textfile):
        with open(textfile, "r") as f:
            sentences = f.readlines()
        sent_proc = list(map(self.field.preprocess, sentences))
        self.field.build_vocab(sent_proc, min_freq=self.min_freq)
        self.len = len(self.field.vocab)
        self.padidx = self.field.vocab.stoi["<pad>"]
        self.bosidx = self.field.vocab.stoi["<bos>"]
        self.eosidx = self.field.vocab.stoi["<eos>"]

    """
    Tokenize and numericalize a batched sentence.
    Converts into torch.Tensor from list of captions.
    Args:
        sentence_batch:     list of str; captions put together in a list
    Returns:
        out:                torch.Tensor;       (batch_size x max_len)
    """

    def encode(self, sentence):
        assert isinstance(sentence, list)
        preprocessed = list(map(self.field.preprocess, sentence))
        out = self.field.process(preprocessed)
        return out

    """
    Reverse conversion from torch.Tensor to a list of captions.
    Args:
        ten:    torch.Tensor (bs x seq)
    Returns:
        out:    list of str
    """

    def decode(self, ten):
        assert isinstance(ten, torch.Tensor)
        assert ten.dtype == torch.long
        assert ten.dim() == 2
        ten = ten.tolist()
        out = []
        for idxs in ten:
            tokenlist = []
            for idx in idxs:
                if idx == self.eosidx:
                    break
                tokenlist.append(self.field.vocab.itos[idx])
            out.append(" ".join(tokenlist))
        return out

    def __len__(self):
        return self.len


def get_captions(jsonfile: str):
    sentences = []
    with open(jsonfile, "r") as f:
        alldata = json.load(f)
    for data in alldata.values():
        sentences.extend(data["sentences"])
    return sentences


# for debugging and creation of text file
if __name__ == "__main__":
    root = "/home/seito/hdd/dsets/activitynet_captions"
    dst = os.path.join(root, "captions/captions.txt")
    # first time only
    if not os.path.exists(dst):
        phases = ["train", "val_1", "val_2"]
        captions = []
        for phase in phases:
            file = os.path.join(root, "captions/{}.json".format(phase))
            captions.extend(get_captions(file))
        with open(dst, "w+") as f:
            f.write("\n".join(captions))
    tokenizer = BasicTokenizer(min_freq=3, max_len=30)
    tokenizer.from_textfile(dst)

    print(len(tokenizer))
    sentence1 = [
        "hello world this is my friend Alex.",
        "the cat and the rat sat on a mat.",
    ]
    ten = tokenizer.encode(sentence1)
    print(ten)
    sent = tokenizer.decode(ten)
    print(sent)
