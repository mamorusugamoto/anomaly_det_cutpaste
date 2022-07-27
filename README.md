# 異常検知モデルCutPasteをmetal_nut画像を1画像ずつ判定するように改良(eval_one_img.py)
(参考)

[CutPaste: Self-Supervised Learning for Anomaly Detection and Localization](https://arxiv.org/abs/2104.04015)

[CutPaste: pythonコード](https://github.com/Runinho/pytorch-cutpaste)

## 実行環境はgoogle colaboratory
google colaboratory上でanomaly_detection_cutpaste.ipynbを開く

以下の２つのファイルを別途ダウンロードして、

google colaboratory上にアップロードする

１、学習済みファイル model-metal_nut-2022-07-09_13_26_16.tch

２、推論時に使う正常な画像 good.zip

## 推論したいmetal_nut画像をgoogle colaboratory上にアップする

--filepath (画像パス)で推論したいmetal_nut画像のパスを指定する

!python eval_one_img.py --filepath Data/002.png --model_dir models  --type metal_nut  --head_layer 2

## 推論コマンド


## 以下は[CutPaste: pythonコード](https://github.com/Runinho/pytorch-cutpaste)でのsetup手順

## Setup
Download the MVTec Anomaly detection Dataset from [here](https://www.mvtec.com/company/research/datasets/mvtec-ad) and extract it into a new folder named `Data`.

Install the following requirements:
1. Pytorch and torchvision
2. sklearn
3. pandas
4. seaborn
5. tqdm
6. tensorboard

For example with [Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html):
```
conda create -n cutpaste pytorch torchvision torchaudio cudatoolkit=10.2 seaborn pandas tqdm tensorboard scikit-learn -c pytorch
conda activate cutpaste
```

## Run Training
```
python run_training.py --model_dir models --head_layer 2
```
The Script will train a model for each defect type and save it in the `model_dir` Folder.

To enable training on an Nvidia GPU use the `--cuda 1` flag.
```
python run_training.py --model_dir models --head_layer 2 --cuda 1
```

One can track the training progress of the models with tensorboard:
```
tensorboard --logdir logdirs
```

## Run Evaluation
```
python eval.py --model_dir models --head_layer 2
```
This will create a new directory `Eval` with plots for each defect type/model.
