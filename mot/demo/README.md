To run tests:
1. Go to mot/
2. Create virtual environment:
```
    python3 -m venv venv
```
3. Activate it:
```
    source venv/bin/activate
```
4. Install torch:
```
    pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```
5. Install mmcv:
```
    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
```
6. Install mmdetection packages:
```
    pip install -e mmdetection
```
7. Install your top level package using pip:
```
    pip install -e .
```
8. Install wget and filterpy:
```
    pip install wget
    pip install filterpy
```

9. Install scipy

```
    pip install scipy
```

10. Install cv2 and seaborn and ffmpeg and ipython

```
    pip install opencv-python
    pip install seaborn
    pip install ffmpeg
    pip install ipython
    pip install ffmpeg-python
    brew install ffmpeg
```


For test_detector.py:
1. Run:
```
    python3 demo/test_detector.py
```
