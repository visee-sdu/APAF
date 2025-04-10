1. The environment setup is the same as that of R2R and R4R.
2. Download datasets of RxR [here](https://huggingface.co/datasets/bowen666212/APAF_RxR/tree/main). It contains the *datasets* and *img_features* folders. Put them in the current directory.
3. Put the object features in the current directory.
4. (Optional). Download the trained model [here](https://huggingface.co/bowen666212/APAF_trained_model/tree/main)

## Pre-training
```
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/pt_rxr.bash 8001
```
## Fine-tuning
```
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/ft_rxr.bash 8001
```
