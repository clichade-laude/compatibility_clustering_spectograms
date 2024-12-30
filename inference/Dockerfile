FROM python:3.10.15

WORKDIR /home/app

RUN pip install prometheus_client torch torchvision numpy 

COPY inference/inference.py .
COPY utils utils

ENTRYPOINT ["python3", "inference.py"]

