FROM cnstark/pytorch:1.7.1-py3.9.12-ubuntu18.04

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
COPY descriptors/ .
COPY dataset/ .

ENV PORT=${PORT:-5000}
EXPOSE ${PORT}

WORKDIR /app
CMD gunicorn --bind 0.0.0.0:$PORT app:app