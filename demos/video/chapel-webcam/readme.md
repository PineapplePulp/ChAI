

```
cd ChAI/build
make bridge_objs
cd demos/video/chapel-webcam
make cleanall && make clean && make libsmol && make main
./main --modelPath sobel.pt

./main --chaiImpl=true --accelScale=0.45 --modelPath sobel.pt
```