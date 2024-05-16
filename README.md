This repo is for using KANs in text classification over GLUE tasks (WNLI, RTE, COLA, MRPC, etc). Our paper will be published in arXiv soon.

# Requirements
* Python >= 3.9
* Install pykan (https://github.com/KindXiaoming/pykan)

# Training

We use **bert-base-cased** as the pre-trained model for producing embeddings (pooled_outputs) in the training process. All models have 768 input size, 64 hidden neurons, and 2 output classes (0 & 1). The training was performed  on 10 epochs with lr = 1e-3.

## EfficientKAN (a modified version of https://github.com/Blealtan/efficient-kan)
```python run_train.py --mode "train" --network "efficient kan" --em_model_name "bert-base-cased" --ds_name "wnli" --epochs 10 --batch_size 4 --max_len 512 --n_size 1 --m_size 768 --n_hidden 64 --n_class 2 --embed_type "pool"```

## TransformerMLP
```python run_train.py --mode "train" --network "mlp" --em_model_name "bert-base-cased" --ds_name "wnli" --epochs 10 --batch_size 4 --max_len 512 --n_size 1 --m_size 768 --n_hidden 64 --n_class 2 --embed_type "pool"```

## TransformerClassifier (with Dropout and Linear)
```python run_train.py --mode "train" --network "mlp" --em_model_name "bert-base-cased" --ds_name "wnli" --epochs 10 --batch_size 4 --max_len 512 --n_size 1 --m_size 768 --n_hidden 64 --n_class 2 --embed_type "pool"```

## Original KAN (https://github.com/KindXiaoming/pykan)
**The training takes a long time to infer outputs from the original KAN model (outputs = KAN(texts)). We are currently working on alternative solutions.**

```python run_train.py --mode "train" --network "kan" --em_model_name "bert-base-cased" --ds_name "wnli" --epochs 10 --batch_size 4 --max_len 512 --n_size 1 --m_size 768 --n_hidden 64 --n_class 2 --embed_type "pool"```

## Parameters
* *mode*: working mode (train or test)
* *network*: type of model (efficientkan, classifier, mlp)
* *em_model_name*: the model offers embeddings (BERT)
* *ds_name*: dataset name
* *epochs*: the number of epochs
* *batch_size*: the training batch size
* *max_len*: the maximum length of input text
* *n_size, m_size*: We consider the input size a matrix with n_size x m_size. For example, BERT offers 768 input size (1 x 768).
* *n_hidden*: The number of hidden neurons. We use only 1 hidden layer. You can modify the code for more layers.
* *n_class*: The number of classes. For GLUE tasks, there are only 2 classes (0 & 1)
* *embed_type*: the type of embeddings (pool, last hidden, or weight)

# Results
Note that the validation accuracy values by tasks have a limit due to the GLUE dataset nature.

## WMLI
| Network  | Best Training Accuracy | Validation Accuracy | Training time (seconds) |
| ------------- | ------------- |  ------------- | ------------- |
| EfficientKAN  |  0.5288 |   0.5648 | **202**  |
| Classifier  |  **0.5414** |   0.5648 | 410  |
| TransformerMLP  | 0.5335 |   0.5648 | 407  |

## MRPC
| Network  | Best Training Accuracy | Validation Accuracy | Training time (seconds) |
| ------------- | ------------- |  ------------- |  ------------- |
| EfficientKAN  |  **0.6782** |  0.6838 | **779**  |
| Classifier  | 0.6712  |   0.6838 | 2015  |
| TransformerMLP  | 0.6744 |   0.6838 | 2012 |


## RTE
| Network  | Best Training Accuracy | Validation Accuracy | Training time (seconds) |
| ------------- | ------------- |  ------------- | ------------- |
| EfficientKAN  |  **0.5248** |  **0.5428** | **547** |
| Classifier  | 0.5016  |   0.5214 | 1379 |
| TransformerMLP  | 0.5016 |   0.5214 | 1392 |

## COLA (5 epochs)
| Network  | Best Training Accuracy | Validation Accuracy | Training time (seconds) |
| ------------- | ------------- |  ------------- | ------------- |
| EfficientKAN  |  - | 0.6912 | |
| Classifier  |   |   0.6912 | |
| TransformerMLP  | 0.7043 |   0.6912 | 2299 |

# References
* https://github.com/Blealtan/efficient-kan
* https://github.com/KindXiaoming/pykan

# Contact
If you have any questions, please contact: tahoangthang@gmail.com

