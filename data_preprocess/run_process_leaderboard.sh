# Process all datasets

set -e

# Math datasets
python data_preprocess/math/dapo_or1_merge_dedup_apr30.py

# Code datasets
python data_preprocess/codegen/humaneval.py
python data_preprocess/codegen/humanevalplus.py
python data_preprocess/codegen/mbpp.py
python data_preprocess/codegen/mbppplus.py
python data_preprocess/codegen/livecodebench.py

# Stem datasets
python data_preprocess/stem/gpqa.py
python data_preprocess/stem/gpqa_diamond.py
python data_preprocess/stem/supergpqa.py


