This repo uses Kolmogorov-Arnold Networks (KANs) for text classification over GLUE tasks (WNLI, RTE, CoLA, MRPC, etc). Our paper will be published in arXiv soon.

# Requirements
* Python >= 3.9
* Install pykan (https://github.com/KindXiaoming/pykan)

# Training

We use **bert-base-cased** as the pre-trained model for producing embeddings (pooled_outputs) in the training process. All models have 768 input size, 64 hidden neurons, and 2 output classes (0 & 1). The training was performed  on 10 epochs with lr = 1e-3.

## EfficientKAN (a modified version of https://github.com/Blealtan/efficient-kan)
```python run_train.py --mode "train" --network "efficientkan" --em_model_name "bert-base-cased" --ds_name "wnli" --epochs 10 --batch_size 4 --max_len 512 --n_size 1 --m_size 768 --n_hidden 64 --n_class 2 --embed_type "pool"```

## FastKAN (https://github.com/ZiyaoLi/fast-kan/)
```python run_train.py --mode "train" --network "fastkan" --em_model_name "bert-base-cased" --ds_name "wnli" --epochs 10 --batch_size 4 --max_len 512 --n_size 1 --m_size 768 --n_hidden 64 --n_class 2 --embed_type "pool"```

## TransformerMLP
```python run_train.py --mode "train" --network "mlp" --em_model_name "bert-base-cased" --ds_name "wnli" --epochs 10 --batch_size 4 --max_len 512 --n_size 1 --m_size 768 --n_hidden 64 --n_class 2 --embed_type "pool"```

## TransformerClassifier (with Dropout and Linear)
```python run_train.py --mode "train" --network "classifier" --em_model_name "bert-base-cased" --ds_name "wnli" --epochs 10 --batch_size 4 --max_len 512 --n_size 1 --m_size 768 --n_hidden 64 --n_class 2 --embed_type "pool"```

## Original KAN (https://github.com/KindXiaoming/pykan)
The training takes **a very long time** when the model infers outputs with an input size of 768 (outputs = KAN(texts)). Therefore, we must reduce the embedding size from 768 to 8 (n_size*m_size) by using reduce_size() in **utils.py**. The smaller the input size, the faster the training time.

```
def reduce_size(embeddings, n_size = 1, m_size = 8):
    second_dim = list(embeddings.shape)[-1]
    first_dim = list(embeddings.shape)[0]
    embeddings = torch.reshape(embeddings, (first_dim, int(second_dim/(n_size*m_size)), n_size*m_size))
    embeddings = torch.sum(embeddings, (1), keepdim = True).squeeze()
    return embeddings
```

Then, we can reluctantly run the training:

```python run_train.py --mode "train" --network "kan" --em_model_name "bert-base-cased" --ds_name "wnli" --epochs 10 --batch_size 4 --max_len 512 --n_size 1 --m_size 8 --n_hidden 64 --n_class 2 --device "cpu"```

## Parameters
* *mode*: working mode ("train" or "test")
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
* *device*: use "cuda" or "cpu"

# Results
Note that Test Accuracy is limited due to the nature of the GLUE datasets. Generally, KANs take less (a half) training time while keeping on-par accuracy compared to TransformerMLPs and TransformerClassifiers. The accuracy values may be better if using embeddings from LLMs.

## WMLI (10 epochs)
| Network  | Best Training Accuracy | Test Accuracy | Training time (seconds) |
| ------------- | ------------- |  ------------- | ------------- |
| EfficientKAN  |  0.5288 |   0.5648 | **202**  |
| Classifier  |  **0.5414** |   0.5648 | 410  |
| TransformerMLP  | 0.5335 |   0.5648 | 407  |
| Original KAN  | 0.5209 |  **0.5925** | 4580  |
| FastKAN  | - | - | - |
| FasterKAN  | - | - | - |

## MRPC (10 epochs)
| Network  | Best Training Accuracy | Test Accuracy | Training time (seconds) |
| ------------- | ------------- |  ------------- |  ------------- |
| EfficientKAN  |  **0.6782** |  0.6838 | **779**  |
| Classifier  | 0.6712  |   0.6838 | 2015  |
| TransformerMLP  | 0.6744 |   0.6838 | 2012 |
| FastKAN  | - | - | - |
| FasterKAN  | - | - | - |


## RTE (10 epochs)
| Network  | Best Training Accuracy | Test Accuracy | Training time (seconds) |
| ------------- | ------------- |  ------------- | ------------- |
| EfficientKAN  |  **0.5248** |  **0.5428** | **547** |
| Classifier  | 0.5016  |   0.5214 | 1379 |
| TransformerMLP  | 0.5016 |   0.5214 | 1392 |
| FastKAN  | - | - | - |
| FasterKAN  | - | - | - |

## CoLA (5 epochs)
| Network  | Best Training Accuracy | Test Accuracy | Training time (seconds) |
| ------------- | ------------- |  ------------- | ------------- |
| EfficientKAN  | **0.7083** | **0.6931** | **866** |
| Classifier  | 0.6915  |   0.6912  | 2286 |
| TransformerMLP  | 0.7043 |   0.6912 | 2299 |
| FastKAN  | - | - | - |
| FasterKAN  | - | - | - |

# References
* https://github.com/Blealtan/efficient-kan
* https://github.com/KindXiaoming/pykan

# Contact
If you have any questions, please contact: tahoangthang@gmail.com

