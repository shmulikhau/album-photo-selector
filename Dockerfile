FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

WORKDIR /app

COPY requirements.txt /requirements.txt

RUN pip install -r /requirements.txt
RUN python -c "from PIL import Image;from transformers import AutoModel;\
AutoModel.from_pretrained('jinaai/jina-clip-v2',trust_remote_code=True)\
.encode_image([Image.new('RGB',(50,50))])"

COPY src /app/src
COPY streamlit_app.py /app/streamlit_app.py

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py"]