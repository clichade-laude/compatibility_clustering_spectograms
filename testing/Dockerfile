FROM python:3.10.15

WORKDIR /home/app

RUN pip install numpy torch torchvision paho-mqtt

COPY testing/test.py .
COPY utils utils

CMD ["python3", "test.py"]

