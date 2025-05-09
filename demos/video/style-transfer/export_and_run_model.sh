


MODEL_NAME="oil_h_bt4_5e11_ep4_epoch_0_batch_id_18000"

python3 neural_style.py export --model saved_models/${MODEL_NAME}.pth --accel

python3 style_transfer_test.py --model-file=models/exports/cpu/${MODEL_NAME}_float16.pt --input-video-file=videos/deer.mp4 --show-output