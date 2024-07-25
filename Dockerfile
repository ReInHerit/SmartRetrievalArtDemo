FROM python:3.8.0-slim

# Create a working directory
WORKDIR /app

RUN apt update && DEBIAN_FRONTEND=noninteractive apt install ffmpeg python3-opencv git bzip2 wget unzip -yq

# Install Python packages
COPY source/requirements.txt .

RUN pip3 install --upgrade pip
RUN pip install \
  torch==1.7.1 \
  torchvision==0.8.2
RUN pip3 install -r requirements.txt

COPY source/ .
RUN python3 ./download_model.py
COPY descriptors/ .
COPY dataset/noisyart_dataset/noisyart ./noisyart_dataset/noisyart
COPY dataset/noisyart_dataset/test_200 ./noisyart_dataset/test_200
COPY dataset/noisyart_dataset/trainval_200 ./noisyart_dataset/trainval_200
COPY dataset/noisyart_dataset/trainval_3120_a/ ./noisyart_dataset/trainval_3120
COPY dataset/noisyart_dataset/trainval_3120_b/ ./noisyart_dataset/trainval_3120

ENV PORT=${PORT:-5000}
EXPOSE ${PORT}

WORKDIR /app
CMD gunicorn --bind 0.0.0.0:$PORT app:app