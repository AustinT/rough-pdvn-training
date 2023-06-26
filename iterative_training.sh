# bash script to run iterative PDVN training

# Start by setting default values to key variables
# ================================================

if [[ -z $output_dir ]]; then
    echo "output_dir must be set"
    exit 1
fi

if [[ -z $input_smiles_file ]]; then
    echo "input_smiles_file must be set"
    exit 1
fi

# Number of outer iterations to run
if [[ -z $num_plan_train_iter ]]; then
    echo "num_plan_train_iter not set, setting default"
    num_plan_train_iter=10
fi

# Molecules to sample for planning in each iteration
if [[ -z $num_mols_per_iter ]]; then
    echo "num_plan_train_iter not set, setting default"
    num_mols_per_iter=2048
fi

# Number of searches to run in parallel
if [[ -z $num_parallel_searches ]]; then
    echo "num_parallel_searches not set, setting default"
    num_parallel_searches=2
fi

# Number of iterations in each MCTS search
if [[ -z $num_mcts_iter ]]; then
    echo "num_mcts_iter not set, setting default"
    num_mcts_iter="10_000"  # could increase later
fi

# Number policy training epochs
if [[ -z $num_policy_train_epochs ]]; then
    echo "num_policy_train_epochs not set, setting default"
    num_policy_train_epochs=2
fi
    

# Run the iterative training / planning
# =====================================
last_policy_checkpoint=""
for (( i=1; i<=$num_plan_train_iter; i++ )) ; do
    echo "Start plan + train iteration $i"
    
    # Make log dir for this iteration
    curr_iter_dir="$output_dir/iter_$i"
    mkdir -p "$curr_iter_dir"

    # Sample random batch of molecules (*without* replacement)
    mol_file_this_batch="$curr_iter_dir/random_mols_all.smi"
    python sample_random_molecules.py \
        --input_file "$input_smiles_file" \
        --num_samples "$num_mols_per_iter" \
        --output_file "$mol_file_this_batch"
    
    # Split this file into chunks for parallel workers
    # NOTE: using `split` does not guarantee that the number of lines in each chunk is the same,
    # but it should be roughly the same. Good enough for a first version.
    mol_file_chunks_dir="$curr_iter_dir/mol_file_chunks"
    mkdir -p "$mol_file_chunks_dir"
    split --number=l/$num_parallel_searches "$mol_file_this_batch" "$mol_file_chunks_dir/mol_file_chunk_" --additional-suffix=.smi

    # Run parallel searches simultaneously
    echo "Starting parallel searches iter $i"
    for mol_file in "$mol_file_chunks_dir"/*.smi ; do
        search_results_this_iter_dir="$curr_iter_dir/search_results/"$(basename "$mol_file" .smi)
        mkdir -p "$search_results_this_iter_dir"
        
        # NOTE: logging command outputs to background
        python run_pdvn_mcts.py \
            --input_file="$mol_file" \
            --output_dir="$search_results_this_iter_dir" \
            --policy_checkpoint="$last_policy_checkpoint" \
            --num_iters=$num_mcts_iter > "$search_results_this_iter_dir.log" &
    done
    wait  # ensure all searches have finished before proceeding

    # Combine results from all searches into a single directory.
    # Necessary because polciy training script only accepts a single directory
    all_results_dir="$curr_iter_dir/search_results/COMBINED"  # important: this dir does not exist yet so is not used in loop below
    for mol_file_dir in "$curr_iter_dir/search_results/"* ; do
        if [[ -d "$mol_file_dir" ]]; then
            # This is a directory. Move all pkl files to single search results dir,
            # being careful not to overwrite anything
            mkdir -p "$all_results_dir"
            for pkl_file in "$mol_file_dir"/*.pkl ; do
                mv "$pkl_file" "$all_results_dir/"$(basename "$mol_file_dir")"__"$(basename "$pkl_file")
            done
        fi
    done

    # Update policy network
    echo "Search complete. Starting policy training iter $i"
    policy_checkpoint_dir="$curr_iter_dir/policy_training_checkpoints"
    mkdir -p "$policy_checkpoint_dir"
    python train_policy_mlp.py \
        --search_result_dir="$all_results_dir" \
        --output_dir="$policy_checkpoint_dir" \
        --starting_checkpoint="$last_policy_checkpoint" \
        --num_epochs="$num_policy_train_epochs"  > "$policy_checkpoint_dir/training.log"
    
    # Update last policy checkpoint
    last_policy_checkpoint=$(ls "$policy_checkpoint_dir"/*.pt | tail -n 1)
    echo "Policy training for iter $i complete. Latest policy checkpoint: $last_policy_checkpoint"

done
