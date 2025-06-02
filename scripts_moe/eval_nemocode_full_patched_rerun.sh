#!/bin/bash

cd ..

# Define single variables (not lists)
train_data="nemocode"
method="full"

# Define lists of variables to iterate through
bs_list=(
    "bs64"
    # "bs32"
)

lr_list=(
    "lr1e-5"
    # "lr3e-5"
    # "lr5e-5"
    # "lr5e-6"
)

# ----------------- Patch: S1 - Full - lr1e-5 - 0609 - Humaneval & Humaneval+ +& LiveCodeBench -------------

temp_list=(
    "0609"
    # "0708"
    # "1007"
)

benchmark_list=(
    "humaneval.sh"
    "humanevalplus.sh"
    # "mbpp.sh"
    # "mbppplus.sh"
    "livecodebench.sh"
)

# Counter for tracking experiments
total_experiments=0
successful_experiments=0
failed_experiments=0

# Iterate through combinations
for bs in "${bs_list[@]}"; do
    for lr in "${lr_list[@]}"; do
        for temp in "${temp_list[@]}"; do
            for benchmark in "${benchmark_list[@]}"; do
                ((total_experiments++))
                
                script_path="scripts_configs/$train_data/$method/$bs/$lr/$temp/$benchmark"
                echo "[$total_experiments] Running: $script_path"
                
                if [ -f "$script_path" ]; then
                    if bash "$script_path"; then
                        ((successful_experiments++))
                        echo "✅ SUCCESS: $script_path"
                    else
                        ((failed_experiments++))
                        echo "❌ FAILED: $script_path"
                    fi
                else
                    ((failed_experiments++))
                    echo "⚠️  MISSING: $script_path"
                fi
                
                echo "----------------------------------------"
            done
        done
    done
done

# ----------------- Patch: S1 - Full - lr1e-5 - 0708 - Humaneval & Humaneval+ & MBPP & LiveCodeBench -------------

temp_list=(
    # "0609"
    "0708"
    # "1007"
)

benchmark_list=(
    "humaneval.sh"
    "humanevalplus.sh"
    "mbpp.sh"
    # "mbppplus.sh"
    "livecodebench.sh"
)

# Counter for tracking experiments
total_experiments=0
successful_experiments=0
failed_experiments=0

# Iterate through combinations
for bs in "${bs_list[@]}"; do
    for lr in "${lr_list[@]}"; do
        for temp in "${temp_list[@]}"; do
            for benchmark in "${benchmark_list[@]}"; do
                ((total_experiments++))
                
                script_path="scripts_configs/$train_data/$method/$bs/$lr/$temp/$benchmark"
                echo "[$total_experiments] Running: $script_path"
                
                if [ -f "$script_path" ]; then
                    if bash "$script_path"; then
                        ((successful_experiments++))
                        echo "✅ SUCCESS: $script_path"
                    else
                        ((failed_experiments++))
                        echo "❌ FAILED: $script_path"
                    fi
                else
                    ((failed_experiments++))
                    echo "⚠️  MISSING: $script_path"
                fi
                
                echo "----------------------------------------"
            done
        done
    done
done

# ----------------- Patch: S1 - Full - lr1e-5 - 1007 - Humaneval & Humaneval+ & MBPP+ & LiveCodeBench -------------

temp_list=(
    # "0609"
    "0708"
    # "1007"
)

benchmark_list=(
    "humaneval.sh"
    "humanevalplus.sh"
    # "mbpp.sh"
    "mbppplus.sh"
    "livecodebench.sh"
)

# Counter for tracking experiments
total_experiments=0
successful_experiments=0
failed_experiments=0

# Iterate through combinations
for bs in "${bs_list[@]}"; do
    for lr in "${lr_list[@]}"; do
        for temp in "${temp_list[@]}"; do
            for benchmark in "${benchmark_list[@]}"; do
                ((total_experiments++))
                
                script_path="scripts_configs/$train_data/$method/$bs/$lr/$temp/$benchmark"
                echo "[$total_experiments] Running: $script_path"
                
                if [ -f "$script_path" ]; then
                    if bash "$script_path"; then
                        ((successful_experiments++))
                        echo "✅ SUCCESS: $script_path"
                    else
                        ((failed_experiments++))
                        echo "❌ FAILED: $script_path"
                    fi
                else
                    ((failed_experiments++))
                    echo "⚠️  MISSING: $script_path"
                fi
                
                echo "----------------------------------------"
            done
        done
    done
done
