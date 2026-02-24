
# prepare data for training [mocked]

# python recipe/swe_agent/prepare/prepare_data.py \
#     --mode simple \
#     --train_size 8 \
#     --test_size 2 \
#     --output_dir data/swe_agent_test

# prepare data for training [real]
### swe-bench [train & test]

echo "prepare data for swe-bench [train & test]"

python recipe/swe_agent/prepare/prepare_data.py \
    --mode swebench \
    --swebench_train /cfs_turbo/yuleiqin/datasets/SWE-bench/data/train-00000-of-00001.jsonl \
    --swebench_test /cfs_turbo/yuleiqin/datasets/SWE-bench/data/test-00000-of-00001.jsonl \
    --output_dir data/swe_agent_swebench

### swe-bench lite [dev & test]

echo "prepare data for swe-bench lite [dev & test]"

python recipe/swe_agent/prepare/prepare_data.py \
    --mode swebench \
    --swebench_train /cfs_turbo/yuleiqin/datasets/SWE-bench_Lite/data/dev-00000-of-00001.jsonl \
    --swebench_test /cfs_turbo/yuleiqin/datasets/SWE-bench_Lite/data/test-00000-of-00001.jsonl \
    --output_dir data/swe_agent_swebench_lite

### swe-bench verified 

echo "prepare data for swe-bench verified"

python recipe/swe_agent/prepare/prepare_data.py \
    --mode swebench \
    --swebench_test /cfs_turbo/yuleiqin/datasets/SWE-bench_Verified/data/test-00000-of-00001.jsonl \
    --output_dir data/swe_agent_swebench_verified

### swe-bench smith

echo "prepare data for swe-bench smith"

python recipe/swe_agent/prepare/prepare_data.py \
    --mode swebench \
    --swebench_train /cfs_turbo/yuleiqin/datasets/SWE-smith/data/train-all.jsonl \
    --output_dir data/swe_agent_swesmith


### swe-smith train
# /cfs_turbo/yuleiqin/datasets/SWE-smith/data/train-all.jsonl

### swe-bench lite
# /cfs_turbo/yuleiqin/datasets/SWE-bench_Lite/data/dev-00000-of-00001.jsonl
# /cfs_turbo/yuleiqin/datasets/SWE-bench_Lite/data/test-00000-of-00001.jsonl

### swe-bench verified
# /cfs_turbo/yuleiqin/datasets/SWE-bench_Verified/data/test-00000-of-00001.jsonl

### swe-bench train
# /cfs_turbo/yuleiqin/datasets/SWE-bench/data/train-00000-of-00001.jsonl
# /cfs_turbo/yuleiqin/datasets/SWE-bench/data/dev-00000-of-00001.jsonl
# /cfs_turbo/yuleiqin/datasets/SWE-bench/data/test-00000-of-00001.jsonl


