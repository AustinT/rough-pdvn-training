# Script to read in molecules from a SMILES file and sample them
import argparse

import numpy as np

if __name__ == "__main__":

    # Add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        help="Path to input file containing SMILES strings.",
        required=True,
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Path to output directory for generated data.",
        required=True,
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        required=True,
        help="Number of molecules to sample.",
    )
    args = parser.parse_args()

    # Read in molecules
    with open(args.input_file, "r") as f:
        smiles_list = f.read().splitlines()
    
    # Sample molecules
    sampled_smiles = np.random.choice(smiles_list, size=args.num_samples, replace=False)

    # Write to file
    with open(args.output_file, "w") as f:
        for smiles in sampled_smiles:
            f.write(smiles + "\n")
