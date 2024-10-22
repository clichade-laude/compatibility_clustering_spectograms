FROM python:3.10.15

WORKDIR /home/app

RUN pip install numpy numba torch torchvision

COPY clustering .
COPY utils utils
COPY model model

ENTRYPOINT ["python3", "cluster.py"]

