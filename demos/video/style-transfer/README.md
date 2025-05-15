


python3 neural_style.py export --model saved_models/udnie.pth --accel
python3 neural_style.py export --model saved_models/candy.pth --accel
python3 neural_style.py export --model saved_models/mosaic.pth --accel

python3 style_transfer_test.py --model-file=models/exports/cpu/mosaic_float16.pt --input-video-file=videos/deer.mp4 --output-video-file=videos/mosaic_deer.mp4 --show-output



sh export_and_run_model.sh nature_oil_painting_ep3_bt4_sw3e10_cw_1e5