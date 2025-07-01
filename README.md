# image-selector



## Quick-Start
### Hardware Requirements
#### Minimum
- 8G RAM with GPU, 32 without GPU
- GPU - minimum 8G VRAM
#### Recommended
- Intel i7 or Amd Ryzen 7
- 16 RAM
- Nvidia GeForce RTX 5060 8G-VRAM
### Install/Run
#### Using docker:

cd into project directory:
```cd project_folder```

build the project:
```docker build -t image_selector .```

run with supported gpu:
```docker run -d --gpus all -e USE_GPU=True -e BATCH_SIZE=8 -p 8501:8501 image_selector```

run without supported gpu:
```docker run -d -p 8501:8501 image_selector```

### Start Guide
Open browser, enter to [localhost:8501](localhost:8501)
