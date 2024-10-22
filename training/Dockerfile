FROM python:3.10.15

WORKDIR /home/app

RUN pip install numpy torch torchvision

COPY training/train.py .
COPY utils utils
COPY model model

ENTRYPOINT ["python3", "train.py"]

