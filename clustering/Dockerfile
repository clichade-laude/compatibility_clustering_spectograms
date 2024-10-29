FROM python:3.10.15

WORKDIR /home/app

RUN pip install numpy numba torch torchvision paho-mqtt

COPY clustering clustering
COPY utils utils

CMD ["python3", "-m", "clustering.cluster"]

