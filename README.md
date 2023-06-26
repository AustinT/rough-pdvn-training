# Rough PDVN training

First attempt at writing a training loop for PDVN on the retro* task.

## Installation

Requires:
- `syntheseus` on the branch with PDVN
- [Pre-trained reaction models](https://github.com/AustinT/pretrained-reaction-models/) to be in the `PYTHONPATH` (to access the retro\* model). Also, all the retro\* files should be downloaded.
- `pytorch` (to run reaction model from retro*)


## Running

Code can be run with a series of 3 scripts.

### Script 1: generating training data

The first script generates training data for MCTS by running search on a sequence of input molecules.

It can be run with:

```bash
python run_pdvn_mcts.py --input_file=path/to/smiles.txt --output_dir=./search_results/run1 --num_iters=100_000
```

Change any arguments to customize the SMILES which are run and the length of the MCTS.
The output of this script will be a series of result files `result_0.pkl`, `result_1.pkl`, ...

### Script 2: training a policy model

This script reads the training data and trains a policy model.
The policy model is just a fine-tuned version of the original retro* reaction model.

To run training, use:

```bash
python train_policy_mlp.py --search_result_dir=./search_results/run1 --output_dir=./training_checkpoints/run1
```

### Script 3: run planning with trained policy model

This script runs retro*-0 using the trained policy model.

```bash
python run_retro_star0_with_policy.py --checkpoint_path=./training_checkpoints/run1/checkpoint-epoch-0.pt  # change checkpoint as desired
```