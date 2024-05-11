**Run the code**

python tran_kan.py --mode "train" --em_model_name "bert-base-cased" --ds_name "wnli" --epochs 10 --batch_size 4 --max_len 512 --n_size 1 --m_size 768 --n_hidden 64 --n_class 2

* ds_name: dataset name
* this KAN have only 1 hidden layer (768, 64, 2), 768 BERT input, 64 hidden neurons, 2 output classes

Note: 
* kan.py: https://github.com/Blealtan/efficient-kan
* https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
