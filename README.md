# image-selector

## Quick-Start
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

### start guide
