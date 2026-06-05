# infer
EXPERIMENT=exp7_480000

CUDA_VISIBLE_DEVICES=6 \
python inference_all_images.py \
--in_path  ./dataset/test/lr \
--out_path  ./results/${EXPERIMENT}/images \
--scale 4 \
--task classical \
--checkpoint experiments/exp1/models/net_g_480000.pth

# gen csv
echo "Training experiment: ${EXPERIMENT}"
CUDA_VISIBLE_DEVICES=6 \
python gen_submission.py \
--folder ./results/${EXPERIMENT}/images \
--save-path ./results/${EXPERIMENT}/${EXPERIMENT}.csv \
--public-size 1

# submit to kaggle
kaggle competitions submit -c super-resolution-in-video-games -f ./results/${EXPERIMENT}/${EXPERIMENT}.csv  -m "${EXPERIMENT}"





# ensemble results
EXPERIMENT=exp7
python ensemble.py \
--experiments exp1 exp7_480000 exp7_490000 \
--out_path ./results/${EXPERIMENT}/images

# gen csv
echo "Training experiment: ${EXPERIMENT}"
CUDA_VISIBLE_DEVICES=6 \
python gen_submission.py \
--folder ./results/${EXPERIMENT}/images \
--save-path ./results/${EXPERIMENT}/${EXPERIMENT}.csv \
--public-size 1

# submit to kaggle
kaggle competitions submit -c super-resolution-in-video-games -f ./results/${EXPERIMENT}/${EXPERIMENT}.csv  -m "${EXPERIMENT}"