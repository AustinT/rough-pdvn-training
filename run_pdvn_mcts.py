"""Script to generate data with retro* reaction model for PDVN algorithm."""

import argparse
from pathlib import Path
import pickle
import numpy as np

from tqdm import tqdm
import torch
import torch.nn.functional as F

from syntheseus.search.chem import Molecule
from syntheseus.search.algorithms.pdvn import PDVN_MCTS, pdvn_extract_training_data
from syntheseus.search.node_evaluation.base import NoCacheNodeEvaluator
from syntheseus.search.node_evaluation.common import ConstantNodeEvaluator
from retro_star_task import retro_star_inventory, backward_model as retro_star_model
from retro_star_task.retro_star_code.mlp_inference import MLPModel, preprocess as smiles_to_fp

class PretrainedRetroStarModelPolicy(NoCacheNodeEvaluator):
    """MCTS policy which uses the "score" metadata field from the reaction model."""

    def _evaluate_nodes(self, nodes, graph=None) -> list[float]:
        return [n.reaction.metadata["score"] for n in nodes]


class UpdatedRetroStarModelPolicy(NoCacheNodeEvaluator):
    """Policy using a custom checkpoint of the retro* reaction model."""

    def __init__(self, checkpoint_path: str):
        super().__init__()

        self.model = MLPModel(retro_star_model.file_names.RXN_MODEL_CHECKPOINT, retro_star_model.file_names.TEMPLATES, device=-1)
        self.model.net.load_state_dict(torch.load(checkpoint_path))
        self.rule_to_idx = {v: k for k, v in self.model.idx2rules.items()}

    def _evaluate_nodes(self, nodes, graph=None) -> list[float]:

        if len(nodes) == 0:
            return []

        # Get predictions from model
        fps = np.array([smiles_to_fp(n.reaction.product.smiles, self.model.fp_dim) for n in nodes])
        fps = torch.tensor(fps, dtype=torch.float32)
        preds = F.softmax(self.model.net(fps), dim=1)

        # Get scores
        softmax_scores = [preds[i][self.rule_to_idx[n.reaction.metadata["template"]]].item() for i, n in enumerate(nodes)]
        softmax_scores = np.clip(np.asarray(softmax_scores), 1e-3, 0.999)
        return softmax_scores.tolist()

class UpdatedRetroStarRxnModelCost(UpdatedRetroStarModelPolicy):
    """Like model above, except it returns *costs* instead of *scores*"""
    def _evaluate_nodes(self, nodes, graph=None) -> list[float]:
        softmax_scores = np.asarray(super()._evaluate_nodes(nodes, graph))
        costs = -np.log(softmax_scores)
        return costs.tolist()

def main():

    # Add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        help="Path to input file containing SMILES strings.",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to output directory for generated data.",
        required=True,
    )
    parser.add_argument(
        "--num_iters",
        type=int,
        default=100_000,
        help="Number of iterations to run the MCTS algorithm.",
    )
    parser.add_argument(
        "--policy_checkpoint",
        type=str,
        help="Path to policy checkpoint file. If blank, then retro* model is used.",
        required=True,
    )
    args = parser.parse_args()

    # Set up algorithm, reaction model, etc
    if args.policy_checkpoint:
        policy = UpdatedRetroStarModelPolicy(checkpoint_path=args.policy_checkpoint)
    else:
        policy = PretrainedRetroStarModelPolicy()
    pdvn = PDVN_MCTS(
        c_dead=10.0,
        value_function_syn=ConstantNodeEvaluator(0.8),
        value_function_cost=ConstantNodeEvaluator(2.0),
        and_node_cost_fn=ConstantNodeEvaluator(1.0),
        mol_inventory=retro_star_inventory.RetroStarInventory(),
        reaction_model=retro_star_model.RetroStarReactionModel(),
        policy=policy,
        limit_iterations=args.num_iters,
        bound_constant=1e3,  # very exploratory
        prevent_repeat_mol_in_trees=True,
        expand_purchasable_mols=False,
        max_expansion_depth=20,  # i.e. 10 reactions deep
    )

    # Run algorithm on each input SMILES string
    with open(args.input_file, "r") as f:
        smiles_list = f.read().splitlines()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)  # ensure output dir exists
    for i, smiles in enumerate(tqdm(smiles_list)):
        
        # Run search and extract training data
        pdvn.reset()  # try to avoid OOM issues?
        output_graph, _ = pdvn.run_from_mol(Molecule(smiles.strip()))
        training_data = pdvn_extract_training_data(output_graph)

        # Write data to pickle file
        with open(f"{args.output_dir}/result_{i}.pkl", "wb") as f_out:
            pickle.dump(training_data, f_out)

        # Log results from this iter
        print(
            f"SMILES {i}: solved = {output_graph.root_node.has_solution}, "
            f"min cost = {output_graph.root_node.data['pdvn_min_syn_cost']}, "
            f"num nodes = {len(output_graph)}.",
            flush=True,
        )
        
        del output_graph, training_data  # clear memory for next iter

if __name__ == "__main__":
    main()