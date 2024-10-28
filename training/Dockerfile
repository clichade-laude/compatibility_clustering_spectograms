FROM python:3.10.15

WORKDIR /home/app

RUN pip install numpy torch torchvision paho-mqtt

COPY training/train.py .
COPY utils utils
COPY model model

CMD ["python3", "train.py"]

