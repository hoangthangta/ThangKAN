This repo is for using KANs in text classification problems over GLEU datasets (WNLI, RTE, COLA, MRPC, etc).

# Parameters
* mode: working mode (train or test)
* network: type of model (efficientkan, classifier, mlp)
* em_model_name: the model offers embeddings (BERT)
* ds_name: dataset name
* epochs: the number of epochs
* batch_size: the training batch size
* max_len: the maximum length of input text
* n_size, m_size: We consider the input size a matrix with n_size x m_size. When using BERT, we have the input size of 768 (1 x 768).
* n_hidden: The number of hidden neurons. We use only 1 hidden layer, you can modify the code for more layers.
* n_class: The number of classes. For GLUE tasks, there are only 2 classes, 0 and 1.

# Train
We use **bert-base-cased** as the pre-trained model for feeding embeddings in the training process. All models have 768  input size, 64 hidden neurons, and 2 output classes (0 & 1).

```python run_train.py --mode "train" --network "mlp" --em_model_name "bert-base-cased" --ds_name "wnli" --epochs 10 --batch_size 4 --max_len 512 --n_size 1 --m_size 768 --n_hidden 64 --n_class 2```

# References
* https://github.com/Blealtan/efficient-kan
* https://github.com/KindXiaoming/pykan
