# APAF

## Prerequisites
1. Our model was trained and evaluated using the following package dependencies:
* Pytorch 1.9.1
* Python 3.6.12

2. Install Matterport3D simulators: follow instructions [here](https://github.com/peteanderson80/Matterport3DSimulator).

3. Download object features [here](https://huggingface.co/datasets/bowen666212/APAF_object_features).

4. Download datasets of R2R and R4R [here](https://huggingface.co/datasets/bowen666212/APAF_R2R_R4R). It contains a *datasets* folder.

5. (Optional). Download the trained model [here](https://huggingface.co/bowen666212/APAF).

## Pre-training
```
cd pretrain_src
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_r4r.sh 8001  # R4R
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_r2r.sh 8001  # R2R
```
## Fine-tuning
```
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/r4r_b16.sh 8001  # R4R
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/r2r_b16.sh 8001  # R2R
```

## RxR
Please see `APAF_RxR/README.md`.

## Acknowledgement
Codebase from [ScaleVLN](https://github.com/wz0919/ScaleVLN), [BEVBert](https://github.com/MarSaKi/VLN-BEVBert) and [DUET](https://github.com/cshizhe/VLN-DUET).
