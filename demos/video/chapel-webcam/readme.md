

```
cd ChAI/build
make bridge_objs
cd demos/video/chapel-webcam
make cleanall && make clean && make libsmol && make main
./main --modelPath sobel.pt
```