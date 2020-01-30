!/bin/bash
source /home/trace/anaconda3/bin/activate 3d_context

subset_pcts=(10 50 100)
logs_paths=("sn_transfer_10" "sn_transfer_50" "sn_transfer_100")
processed_names=("subset_10" "subset_50" "processed")

for i in {0..3}
do
	echo python -m context.fine_tune_training_sn \
		 --logs_path runs/${logs_paths[i]} \
		 --processed_name ${processed_names[i]} \
		 --load_subset --subset_pct ${subset_pcts[i]} \
		 --model_name "2d_rot_modelnet40_9bands_ckpt_70.pt"
	python -m context.fine_tune_training_sn \
		--logs_path runs/${logs_paths[i]} \
		--processed_name ${processed_names[i]} \
		--load_subset --subset_pct ${subset_pcts[i]} \
		--model_name "2d_rot_modelnet40_9bands_ckpt_70.pt"
done
