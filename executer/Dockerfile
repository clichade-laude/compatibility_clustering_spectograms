FROM python:3.10.15

WORKDIR /home/app

RUN pip install paho-mqtt

COPY executer/mqtt_init.py .
COPY utils/mqtt.py utils/mqtt.py

ENTRYPOINT ["python3", "mqtt_init.py"]

