FROM python:3.10.15

WORKDIR /home/app

RUN pip install paho-mqtt

COPY controller/control_train.py .
COPY utils/mqtt.py utils/mqtt.py

CMD ["python3", "control_train.py"]

