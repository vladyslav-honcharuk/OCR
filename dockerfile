FROM python:3.9

WORKDIR /app

COPY train.py .
COPY inference.py .
COPY model.h5 .
COPY readme.md .
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt 
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y