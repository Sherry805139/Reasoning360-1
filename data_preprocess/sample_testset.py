import pandas as pd

# File paths
reasoning_gym_file = "/mnt/sharefs/users/haonan.li/data/k2/test_12k_len/logic__reasoning_gym_4.3k.parquet"
synlogic_file = "/mnt/sharefs/users/haonan.li/data/k2/test_12k_len/logic__synlogic_1.4k.parquet"

# Load datasets
df_reasoning_gym = pd.read_parquet(reasoning_gym_file)
df_synlogic = pd.read_parquet(synlogic_file)

# Sample reasoning_gym: 5 rows per ability
sampled_reasoning_gym = df_reasoning_gym.groupby('ability').apply(
    lambda x: x.sample(min(5, len(x)), random_state=42)
).reset_index(drop=True)

# Sample synlogic: 10 rows per data_source  
sampled_synlogic = df_synlogic.groupby('data_source').apply(
    lambda x: x.sample(min(10, len(x)), random_state=42)
).reset_index(drop=True)

# Create output filenames with exact numbers
reasoning_gym_output = f"/mnt/sharefs/users/haonan.li/data/k2/test_12k_len/logic__reasoning_gym_{len(sampled_reasoning_gym)}.parquet"
synlogic_output = f"/mnt/sharefs/users/haonan.li/data/k2/test_12k_len/logic__synlogic_{len(sampled_synlogic)}.parquet"

# Save to separate files
sampled_reasoning_gym.to_parquet(reasoning_gym_output, index=False)
sampled_synlogic.to_parquet(synlogic_output, index=False)

print(f"Sampled {len(sampled_reasoning_gym)} from reasoning_gym -> {reasoning_gym_output}")
print(f"Sampled {len(sampled_synlogic)} from synlogic -> {synlogic_output}")
