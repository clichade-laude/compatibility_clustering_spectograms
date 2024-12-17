FROM python:3.10.15

WORKDIR /home/app

RUN pip install paho-mqtt

COPY executer/exec_train.py .
COPY utils/mqtt.py utils/mqtt.py

ENTRYPOINT ["python3", "exec_train.py"]

