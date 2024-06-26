declare -a arr1=("trans_effi_kan" "trans_fast_kan" "trans_faster_kan" "mlp" "classifier" "effi_kan" "fast_kan" "faster_kan")
declare -a arr2=("mrpc" "rte" "cola" "wnli")
for i in "${arr1[@]}"; do for j in "${arr2[@]}";
do
	#echo "$i" "$j"
	python run_train.py --mode "train" --network "$i" --em_model_name "bert-base-cased" --ds_name "$j" --epochs 10 --batch_size 4 --max_len 512 --n_size 1 --m_size 768 --n_hidden 64 --n_class 2 --embed_type "pool"
done;
done
