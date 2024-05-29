This repo uses Kolmogorov-Arnold Networks (KANs) for text classification over GLUE tasks (WNLI, RTE, CoLA, MRPC, etc). Our paper will be published in arXiv soon.

# Requirements
* Python >= 3.9
* Install pykan (https://github.com/KindXiaoming/pykan)
* requirements.txt

# Training

We use **bert-base-cased** as the pre-trained model for producing embeddings (pooled_outputs) in the training process. All models have 768 input size, 64 hidden neurons, and 2 output classes (0 & 1). The training was performed  on xx epochs with lr = 2e-5.

## TransformerEfficientKAN 
```python run_train.py --mode "train" --network "trans_effi_kan" --em_model_name "bert-base-cased" --ds_name "wnli" --epochs 10 --batch_size 4 --max_len 512 --n_size 1 --m_size 768 --n_hidden 64 --n_class 2```

## TransformerFastKAN
```python run_train.py --mode "train" --network "trans_fast_kan" --em_model_name "bert-base-cased" --ds_name "wnli" --epochs 10 --batch_size 4 --max_len 512 --n_size 1 --m_size 768 --n_hidden 64 --n_class 2```

## TransformerFasterKAN
```python run_train.py --mode "train" --network "trans_faster_kan" --em_model_name "bert-base-cased" --ds_name "wnli" --epochs 10 --batch_size 4 --max_len 512 --n_size 1 --m_size 768 --n_hidden 64 --n_class 2```

## TransformerMLP
```python run_train.py --mode "train" --network "mlp" --em_model_name "bert-base-cased" --ds_name "wnli" --epochs 10 --batch_size 4 --max_len 512 --n_size 1 --m_size 768 --n_hidden 64 --n_class 2```

## TransformerClassifier (with Dropout and Linear)
```python run_train.py --mode "train" --network "classifier" --em_model_name "bert-base-cased" --ds_name "wnli" --epochs 10 --batch_size 4 --max_len 512 --n_size 1 --m_size 768 --n_hidden 64 --n_class 2```

## Original KAN
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
* *network*: type of model (efficientkan, TransformerClassifier, mlp)
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
It's important to mention that WNLI inherently struggles to achieve high validation accuracy. Additionally, KANs face significant challenges in achieving convergence during text classification if they are not trained on top of BERT.

## CoLA (10 epochs)
| Network  | Training Accuracy | Val Accuracy | Training time (seconds) |
| ------------- | ------------- |  ------------- | ------------- |
| mlp | 0.9897 | 0.8282 | 2798 |
| classifier | 0.9619 | 0.8282 | 2802 |
| trans_effi_kan | 0.9635 | **0.8292** | 2827 |
| trans_fast_kan | 0.9949 | 0.8206 | 2831 |
| trans_faster_kan | 0.9756 | 0.8215 | 2818 |
| effi_kan | 0.749 | 0.7458 | 951 |
| fast_kan | 0.7501 | 0.742 | 937 |
| faster_kan | 0.7235 | 0.7315 | 924 |

## MRPC (10 epochs)
| Network  | Training Accuracy | Val Accuracy | Training time (seconds) |
| ------------- | ------------- |  ------------- | ------------- |
| mlp | 0.7377 | 0.8603 | 1195 |
| classifier | 0.9866 | **0.8848** | 1204 |
| trans_effi_kan | 0.9986 | 0.8676 | 1219 |
| trans_fast_kan | 0.9422 | 0.8554 | 1214 |
| trans_faster_kan | 0.9591 | 0.8701 | 1207 |
| effi_kan | 0.6955 | 0.7255 | 407 |
| fast_kan | 0.7009 | 0.7157 | 401 |
| faster_kan | 0.6848 | 0.7059 | 395 |

## RTE (10 epochs)
| Network  | Training Accuracy | Val Accuracy | Training time (seconds) |
| ------------- | ------------- |  ------------- | ------------- |
| mlp | 0.9302 | 0.675 | 821 |
| classifier | 0.8475 | 0.625 | 818 |
| trans_effi_kan | 0.9069 | 0.675 | 826 |
| trans_fast_kan | 0.9394 | 0.6071 | 831 |
| trans_faster_kan | 0.9639 | **0.6964** | 829 |
| effi_kan | 0.5004 | 0.5214 | 277 |
| fast_kan | 0.5269 | 0.5429 | 273 |
| faster_kan | 0.496 | 0.5214 | 269 |

## WNLI (10 epochs)
| Network  | Training Accuracy | Val Accuracy | Training time (seconds) |
| ------------- | ------------- |  ------------- | ------------- |
| mlp | 0.5058 | 0.5648 | 213 |
| classifier | 0.5204 | 0.5648 | 210 |
| trans_effi_kan | 0.4182 | 0.5648 | 212 |
| trans_fast_kan | 0.5252 | 0.5648 | 211 |
| trans_faster_kan | 0.4832 | 0.5648 | 211 |
| effi_kan | 0.5147 | 0.5648 | 73 |
| fast_kan | 0.5372 | 0.5648 | 72 |
| faster_kan | 0.501 | 0.5648 | 71 |

# References
* https://github.com/Blealtan/efficient-kan
* https://github.com/KindXiaoming/pykan
* https://github.com/AthanasiosDelis/faster-kan
* https://github.com/ZiyaoLi/fast-kan/

# Contact
If you have any questions, please contact: tahoangthang@gmail.com. If you want to know more about me, please visit website: https://tahoangthang.com.

