# SAMSA
Official implementation of SAMSA: Shift-And-Mix Self-Attention

## Standalone SAMSA implementation

To apply ChordMixer architecture on your datasets, please use both SAMSA.py, dataloader_utils.py, training_utils.py to ensure the correct batch construction.

## Datasets

Follow the instructions from /experiments/README.md to get the datasets from this paper.
All the datasets should be stored in the experiments/data folder. 

## WANDB
Experiment tracking requires installing Weights and Biases. If you dont have an account, please create a free account on wandb.ai. Afterwards, go to Settings and create an API key. Copy the API key and init wandb using this key.

You need to setup wandb account to see the detailed logging and performance results within length log2-bins. 


To get the reported results, please run:

```
cd SAMSA/experiments
python3 train_adding.py --problem_class adding --problem 200 --model 'samsa' --device_id 0 --wandb %yourusername%
python3 train_adding.py --problem_class adding --problem 1000 --model 'samsa' --device_id 0 --wandb %yourusername%
python3 train_adding.py --problem_class adding --problem 16000 --model 'samsa' --device_id 0 --wandb %yourusername%
python3 train_adding.py --problem_class adding --problem 128000 --model 'samsa' --device_id 0 --wandb %yourusername%

python3 train_genbank.py --problem_class genbank --problem 'Carassius vs. Labeo' --model 'samsa' --device_id 0 --wandb %yourusername%
python3 train_genbank.py --problem_class genbank --problem 'Sus vs. Bos' --model 'samsa' --device_id 0 --wandb %yourusername%
python3 train_genbank.py --problem_class genbank --problem 'Mus vs. Rattus' --model 'samsa' --device_id 0 --wandb %yourusername%
python3 train_genbank.py --problem_class genbank --problem 'Danio vs. Cyprinus' --model 'samsa' --device_id 0 --wandb %yourusername%

python3 train_longdoc.py --problem_class longdoc --problem longdoc --model 'samsa' --device_id 0 --wandb %yourusername%

cd SAMSA
python3 train_listops_pad.py --model 'samsa' --device_id 0 --wandb %yourusername%
python3 train_cifar10.py --model 'samsa' --device_id 0 --wandb %yourusername%
python3 train_pathfinder.py --model 'samsa' --device_id 0 --wandb %yourusername%
python3 train_text.py --model 'samsa' --device_id 0 --wandb %yourusername%
python3 train_retrieval.py --model 'samsa' --device_id 0 --wandb %yourusername%

```
