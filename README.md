# NYCU Selected Topics in Visual Recognition using Deep Learning 2026
# Final project on Super Resolution in Video Games


- Team member: Quang-Thang Le, Dinh-Hung Le, Kevin Wijaya

## Introduction
This repository contains the full training and inference pipeline for PFT.

This repository is based on https://github.com/CVL-UESTC/PFT-SR. If you have any issue with this repo, you can also find solutions in the original one.

## Enviroment setup

- Follow instructions in https://github.com/CVL-UESTC/PFT-SR to set up the enviroment

## Usage

1. Download the dataset, extract it, and place it in the `dataset` folder.
2. Run 
```
python process_data.py
```
to split data to train and val sets
3. Run 
```
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch \
--use-env --nproc_per_node=4 --master_port=1145 \
basicsr/train.py \
-opt options/train/final.yml \
--launcher pytorch
```
to train the model

4. Run 
```
EXPERIMENT=final

# infer
CUDA_VISIBLE_DEVICES=0 \
python inference_all_images_TTA_2.py \
--in_path  ./dataset/test/lr \
--out_path  ./results/${EXPERIMENT}/images \
--scale 4 \
--task classical \
--checkpoint experiments/${EXPERIMENT}/models/net_g_latest.pth

# gen csv
python gen_submission.py \
--folder ./results/${EXPERIMENT}/images \
--save-path ./results/${EXPERIMENT}/${EXPERIMENT}.csv \
--public-size 1

# submit to kaggle
kaggle competitions submit -c super-resolution-in-video-games -f ./results/${EXPERIMENT}/${EXPERIMENT}.csv  -m "${EXPERIMENT}"
```
to infer, gen csv file, and submit this file to kaggle

5. The final results are saved at `results/final`.

## Additional Ensemble with HAT-clean Branch

In addition to the PFT-based pipeline above, we also provide another inference pipeline in the `HAT-clean` branch.

To use it, first checkout to the `HAT-clean` branch:

```bash
git checkout HAT-clean
```

Then follow the instructions in the `README.md` file of that branch to set up the environment and run inference.

After finishing inference with both branches, you should have two folders containing the final super-resolved images. Then, run the ensemble script to combine the two inference results:

```bash
python ensemble.py \
--folder1 path/to/first/inference/images \
--folder2 path/to/second/inference/images \
--out_folder path/to/ensemble/results
```

After generating the ensembled images, create the submission CSV file:

```bash
python gen_submission.py \
--folder path/to/ensemble/results \
--save-path path/to/ensemble_submission.csv \
--public-size 1
```

Finally, submit the ensembled result to Kaggle:

```bash
kaggle competitions submit \
-c super-resolution-in-video-games \
-f path/to/ensemble_submission.csv \
-m "ensemble result"
```


## Performance Snapshot

![alt text](assets/ranking.png)

![ranking emsemble](assets/ranking_ensemble.png)