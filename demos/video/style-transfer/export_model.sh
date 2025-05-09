python3 neural_style.py export --model saved_models/ckpt_epoch_0_batch_id_18000.pth --accel
# python3 style_transfer_test.py --model-file=models/exports/cpu/ckpt_epoch_0_batch_id_18000_float16.pt --use-webcam --show-output

python3 style_transfer_test.py --model-file=models/exports/cpu/ckpt_epoch_0_batch_id_18000_float16.pt --input-video-file=videos/deer.mp4 --show-output