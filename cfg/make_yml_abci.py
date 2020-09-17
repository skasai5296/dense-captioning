import sys, os, glob
import math

import numpy as np

def geometric(start, end, base, dtype=float):
    steps = int(math.log(end / start, base)) + 1
    return np.geomspace(start, end, steps, dtype=dtype)

def linear(start, end, size):
    steps = int((end - start) / size) + 1
    return np.linspace(start, end, steps)

if not os.path.exists("abci"):
    os.mkdir("abci")
else:
    for f in glob.glob("abci/cfg*.yml"):
        os.remove(f)
i = 0
for lr in geometric(1e-5, 1e-2, 10):
    for do in linear(0.0, 0.5, 0.1):
        for ss in linear(0.0, 0.3, 0.1):
            name = f"lr{lr}_do{do}_ss{ss}"
            txt = """name: {}
use_wandb: True

path:
  root: /groups1/gaa50131/datasets/ActivityNet
  feature: "features/resnet"
  annotation: captions
  output: ../../out
  submission_file: submission.json

hyperparam:
  attention: True
  tokenizer:
    min_freq: 5
    max_len: 100
  optimization:
    name: Adam
    lr: {:f}
    beta1: 0.9
    beta2: 0.999
    weight_decay: 0.
  loss_weight:
    XELoss: 1.
  transformer:
    feature_dim: 2048
    input_seqlen: 200
    input_dim: 768
    n_head: 12
    n_layer: 12
    dropout_prob: {:f}
    scheduled_sampling_prob: {:f}
  misc:
    max_epoch: 20
    num_worker: 32
    gpu_ids: [0, 1, 2, 3]
    batch_size: 64
    val_samples: 1000
    # set to 0 if making nondeterministic
    random_seed: 0
""".format(name, lr, do, ss)

            i += 1
            with open(f"abci/cfg{i}.yml", "w") as f:
                f.write(txt)
print(f"made {i} yaml files")
