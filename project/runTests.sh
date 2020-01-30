#!/bin/bash
source /home/trace/anaconda3/bin/activate 3d_context
# subset_pcts=(10 30 50 100)
# logs_paths=("sn_baseline_10" "sn_baseline_30" "sn_baseline_50" "sn_baseline_100")
# processed_names=("subset_10" "subset_30" "subset_50" "processed")
subset_pcts=(30 50 100)
logs_paths=("sn_baseline_30" "sn_baseline_50" "sn_baseline_100")
processed_names=("subset_30" "subset_50" "processed")
#echo $logs_paths[0]

for i in {0..4}
do
	echo python -m context.baseline_train_sn --logs_path runs/${logs_paths[i]} --processed_name ${processed_names[i]} --load_subset --subset_pct ${subset_pcts[i]}
	python -m context.baseline_train_sn --logs_path runs/${logs_paths[i]} --processed_name ${processed_names[i]} --load_subset --subset_pct ${subset_pcts[i]}
done
