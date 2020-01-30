#!/bin/bash
source /home/trace/anaconda3/bin/activate 3d_context
# subset_pcts=(10 30 50 100)
# logs_paths=("sn_baseline_10" "sn_baseline_30" "sn_baseline_50" "sn_baseline_100")
# processed_names=("subset_10" "subset_30" "subset_50" "processed")
subset_pcts=(50 100)
logs_paths=("mn40_baseline_50" "mn40_baseline_100")
processed_names=("subset_50" "processed")

for i in {0..2}
do
	echo python -m context.baseline_train \
		--logs_path runs/${logs_paths[i]} \
		--processed_name ${processed_names[i]} \
		--load_subset --subset_pct ${subset_pcts[i]} \
		--batch_size 16 --modelnet_version 40 \
		--n_epochs 100
	python -m context.baseline_train \
		--logs_path runs/${logs_paths[i]} \
		--processed_name ${processed_names[i]} \
		--load_subset --subset_pct ${subset_pcts[i]} \
		--batch_size 16 --modelnet_version 40 \
		--n_epochs 100
done
