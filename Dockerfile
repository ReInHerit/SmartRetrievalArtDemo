## docker build --build-arg INCLUDE_TRAINVAL=true to include also the trainval parts of NoisyArt dataset

FROM python:3.8.0-slim

# Create a working directory
WORKDIR /app

RUN apt update && DEBIAN_FRONTEND=noninteractive apt install --no-install-recommends ffmpeg python3-opencv git bzip2 wget unzip -yq

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

# Define a build argument
ARG INCLUDE_TRAINVAL=false

# Conditionally copy directories based on the build argument
RUN if [ "$INCLUDE_TRAINVAL" = "true" ]; then \
    cp -r dataset/noisyart_dataset/trainval_200 ./noisyart_dataset/trainval_200 && \
    cp -r dataset/noisyart_dataset/trainval_3120_a/ ./noisyart_dataset/trainval_3120 && \
    cp -r dataset/noisyart_dataset/trainval_3120_b/ ./noisyart_dataset/trainval_3120; \
    fi \

ENV PORT=${PORT:-5000}
EXPOSE ${PORT}

WORKDIR /app
CMD gunicorn --bind 0.0.0.0:$PORT app:app