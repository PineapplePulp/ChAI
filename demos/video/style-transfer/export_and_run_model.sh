

MODEL_NAME=$1

# MODEL_NAME="nature_oil_painting_ep4_bt4_sw1e10_cw_1e5_flash_epoch_1_batch_id_8000"
# MODEL_NAME="nature_oil_painting_ep4_bt4_sw5e9_cw_1e5_flash_epoch_0_batch_id_12000"

python3 neural_style.py export --model saved_models/${MODEL_NAME}.model --accel \
    || python3 neural_style.py export --model saved_models/${MODEL_NAME}.pth --accel

# nature_oil_painting_ep3_bt4_sw3e10_cw_1e5

# python3 style_transfer_test.py --model-file=models/exports/cpu/${MODEL_NAME}_float16.pt --input-video-file=videos/deer.mp4 --show-output

python3 style_transfer_test.py --model-file=models/exports/cpu/${MODEL_NAME}_float16.pt --use-webcam --show-output