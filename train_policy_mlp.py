"""Hard-coded script to train a policy model from the retro* dataset."""

import argparse
from pathlib import Path
import pickle

from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
import torch
import numpy as np

from retro_star_task import backward_model as retro_star_model
from retro_star_task.retro_star_code.mlp_policies import preprocess as smiles_to_fp

# Add arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--search_result_dir",
    type=str,
    help="Path to directory containing search results as .pkl files.",
    required=True,
)
parser.add_argument(
    "--output_dir",
    type=str,
    help="Path to save model checkpoints to.",
    required=True,
)
parser.add_argument(
    "--num_epochs",
    type=int,
    default=10,
    help="Number of epochs to train for.",
)
args = parser.parse_args()


# Extract training data for a policy network and write to file
all_result_files = list(Path(args.search_result_dir).glob("result_*.pkl"))
smiles_list = []
templates = []
for pkl_file in tqdm(sorted(all_result_files[:])):
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)

    for mol, rxn_set in data.mol_to_reactions_for_min_syn.items():
        for rxn in rxn_set:
            smiles_list.append(mol.smiles)
            templates.append(rxn.metadata['template'])

# Train new reaction model policy
rxn_model = retro_star_model.RetroStarReactionModel(device=0)  # hard-coded to use GPU
device = rxn_model.model.net.fc1.weight.device

# Process dataset
import numpy as np
rule_2_idx = dict()
for idx, rule in rxn_model.model.idx2rules.items():
    rule_2_idx[rule] = idx

fps = []
template_idx = []
for s, template in tqdm(zip(smiles_list, templates)):
    fps.append(smiles_to_fp(s, 2048))
    template_idx.append(rule_2_idx[template])
fps = np.asarray(fps, dtype=np.float32)
print("Training dataset size: ", len(fps),) 

# Make data loader
dataloader = DataLoader(
    TensorDataset(torch.as_tensor(fps), torch.as_tensor(template_idx)),
    batch_size=8,
    shuffle=True,
    num_workers=4,
    drop_last=True
)
print("Total number of parameters: ", sum(p.nelement() for p in rxn_model.model.net.parameters()))

# Run training
Path(args.output_dir).mkdir(exist_ok=True, parents=True)
rxn_model.model.net.train()
optimizer = torch.optim.Adam(rxn_model.model.net.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()
for epoch in range(50):
    loss_list = []
    for x, y in tqdm(dataloader):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        output = rxn_model.model.net(x)
        loss = loss_fn(output, y)
        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}: avg loss = {np.average(loss_list):.3f}")
    torch.save(rxn_model.model.net.state_dict(), f"{args.output_dir}/checkpoint-epoch-{epoch}.pt")
