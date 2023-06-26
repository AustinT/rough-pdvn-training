"""Run retro*-0 planning with a pre-trained policy."""

import argparse
import pickle
import numpy as np

from tqdm import tqdm

from syntheseus.search.chem import Molecule
from syntheseus.search.node_evaluation.common import ConstantNodeEvaluator
from retro_star_task import retro_star_inventory, backward_model as retro_star_model
from syntheseus.search.algorithms.best_first.retro_star import RetroStarSearch
from syntheseus.search.analysis.solution_time import get_first_solution_time

from run_pdvn_mcts import UpdatedRetroStarRxnModelCost


def main():
    
    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="Path to checkpoint file.",
        required=True,
    )
    args = parser.parse_args()

    # Set up search algorithm
    retro_star0 = RetroStarSearch(
        mol_inventory=retro_star_inventory.RetroStarInventory(),
        reaction_model=retro_star_model.RetroStarReactionModel(use_cache=False),  # they didn't use cache
        limit_reaction_model_calls=500,
        limit_iterations=1_000_000,
        max_expansion_depth=20,  # prevent overly-deep solutions
        prevent_repeat_mol_in_trees=True,  # original paper did this
        expand_purchasable_mols=False,  # original paper did this
        and_node_cost_fn=UpdatedRetroStarRxnModelCost(checkpoint_path=args.checkpoint_path),
        value_function=ConstantNodeEvaluator(0.0),  # retro-star *zero*
    )
    with open(retro_star_model.file_names.TEST_ROUTES, "rb") as f:
        test_routes = pickle.load(f)
    soln_times = []
    for i, route in enumerate(tqdm(test_routes)): 
        retro_star0.reset()
        output_graph, _  = retro_star0.run_from_mol(Molecule(route[0].split(">>")[0]))

        # Analyze solution time
        for node in output_graph.nodes():
            node.data["analysis_time"] = node.data["num_calls_rxn_model"]
        soln_time = get_first_solution_time(output_graph)
        print(i, soln_time, flush=True)
        soln_times.append(soln_time)
    
    print("Final solution times:")
    for t in [10, ] + list(range(50, 501, 50)):
        print(t, np.average([s <= t for s in soln_times]))
    
if __name__ == "__main__":
    main()