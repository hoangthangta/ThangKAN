This repo is for using KANs in text classification problems over GLUE datasets (WNLI, RTE, COLA, MRPC, etc).

# Requirements
* Python >= 3.9

# Training

We use **bert-base-cased** as the pre-trained model for feeding embeddings in the training process. All models have 768 input size, 64 hidden neurons, and 2 output classes (0 & 1). The training was performed  on 10 epochs with lr = 1e-3.

## EfficientKAN
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
* embed_type: the type of embeddings (pool, last hidden, or weight)

# Results

## WMLI
| Network  | Best Training Accuracy | Validation Accuracy | Training time (seconds) |
| ------------- | ------------- |  ------------- | ------------- |
| EfficientKAN  |  **0.5288** |   0.5648 | 202  |
| Classifier  |  0.5414 |   0.5648 | 410  |
| TransformerMLP  | 0.5083 |   0.5648 | 418  |

## MRPC
| Network  | Best Training Accuracy | Validation Accuracy | Training time (seconds) |
| ------------- | ------------- |  ------------- |  ------------- |
| EfficientKAN  |  0.6782 |  0.6838 | 779  |
| Classifier  | 0.6712  |   0.6838 | 2015  |
| TransformerMLP  | 0.6744 |   0.6838 | 2012 |


## RTE
| Network  | Best Training Accuracy | Validation Accuracy | Training time (seconds) |
| ------------- | ------------- |  ------------- | ------------- |
| EfficientKAN  |  - |  - | |
| Classifier  |   |   - | |
| TransformerMLP  | - |   - | |

## COLA
| Network  | Best Training Accuracy | Validation Accuracy | Training time (seconds) |
| ------------- | ------------- |  ------------- | ------------- |
| EfficientKAN  |  - |  - | |
| Classifier  |   |   - | |
| TransformerMLP  | - |   - | |

# References
* https://github.com/Blealtan/efficient-kan
* https://github.com/KindXiaoming/pykan

# Contact
If you have any questions, please contact tahoangthang@gmail.com

