declare -a arr1=("mlp" "classifier")
declare -a arr2=("mrpc" "rte" "cola" "wnli")
for i in "${arr1[@]}"; do for j in "${arr2[@]}";
do
	#echo "$i" "$j"
	python run_train.py --mode "train" --network "$i" --em_model_name "bert-base-cased" --ds_name "$j" --epochs 5 --batch_size 4 --max_len 512 --n_size 1 --m_size 768 --n_hidden 64 --n_class 2 --embed_type "pool"
done;
done
