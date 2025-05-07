


python3 neural_style.py export --model saved_models/udnie.pth --accel

python3 style_transfer_test.py --model-file=models/exports/cpu/mosaic_float16.pt --input-video-file=videos/deer.mp4 --output-video-file=videos/mosaic_deer.mp4 --show-output

