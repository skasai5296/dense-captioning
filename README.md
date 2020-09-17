# actnetchallenge: Task 3 (Dense-Captioning Events in Videos)
Repo for activity net challenge 2020: Task 3 (Dense-Captioning Events in Videos)
This repository provides a dense video captioning module for ActivityNet Captions Dataset.

TO-DO:
- [x] add training of baseline model
- [x] add validation
- [x] add test (evaluation using ground truth proposals)
- [ ] add BERT training
- [ ] add character level training

## Requirements
* Python>=3.7
* wandb (for experiments and logging)
* numpy
* pytorch>=1.0
* torchvision>=0.2
* torchtext, spacy
* pyyaml, addict (for loading configuration files)
* nlg-eval (for evaluation metrics)

## How to download ActivityNet Captions Dataset (ActivityNet Videos + Annotations)
1. Download annotation file for ActivityNet dataset from [here](http://ec2-52-11-11-89.us-west-2.compute.amazonaws.com/files/activity_net.v1-3.min.json)
1. Download annotation files for ActivityNet Captions dataset from [here](https://cs.stanford.edu/people/ranjaykrishna/densevid/captions.zip)

## Training procedures
```bash
cd src
python train.py --config ../cfg/${config_path}.yml --resume
```

## Testing procedures (for validation)
1. Proposal generation is not implemented yet, so prepare an annotation file with proposals.
```bash
cd src
python make_submission.py --config ../cfg/${config_path}.yml
```

### Transformer Captions
![Transformer Captions](assets/transformer_sample.png)
