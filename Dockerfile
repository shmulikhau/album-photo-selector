FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

COPY requirements.txt /requirements.txt

RUN pip install -r /requirements.txt

COPY src /src