# infer
EXPERIMENT=exp11
CUDA_VISIBLE_DEVICES=2 \
python inference_all_images.py \
--in_path  ./dataset/test/lr \
--out_path  ./results/${EXPERIMENT}/images \
--scale 4 \
--task classical \
--checkpoint experiments/${EXPERIMENT}/models/net_g_100000.pth

# gen csv
echo "Training experiment: ${EXPERIMENT}"
python gen_submission.py \
--folder ./results/${EXPERIMENT}/images \
--save-path ./results/${EXPERIMENT}/${EXPERIMENT}.csv \
--public-size 1

# submit to kaggle
kaggle competitions submit -c super-resolution-in-video-games -f ./results/${EXPERIMENT}/${EXPERIMENT}.csv  -m "${EXPERIMENT}"



##########################################
##########################################
# Infer TTA ver 2
EXPERIMENT=exp11

CUDA_VISIBLE_DEVICES=7 \
python inference_all_images_TTA_2.py \
--in_path  ./dataset/test/lr \
--out_path  ./results/${EXPERIMENT}/images \
--scale 4 \
--task classical \
--checkpoint experiments/${EXPERIMENT}/models/net_g_latest.pth

# gen csv
echo "Training experiment: ${EXPERIMENT}"

python gen_submission.py \
--folder ./results/${EXPERIMENT}/images \
--save-path ./results/${EXPERIMENT}/${EXPERIMENT}.csv \
--public-size 1

# submit to kaggle
kaggle competitions submit -c super-resolution-in-video-games -f ./results/${EXPERIMENT}/${EXPERIMENT}.csv  -m "${EXPERIMENT}"


##########################################
##########################################
# Ensemble
python ensemble.py -e exp9 exp11 -o results/ensemble_9_11
kaggle competitions submit -c super-resolution-in-video-games -f ./results/ensemble_9_11/ensemble_9_11.csv  -m "ensemble_9_11"