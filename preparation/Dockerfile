FROM python:3.10.15

WORKDIR /home/app

RUN pip install numpy

COPY preparation preparation

CMD ["./preparation/download.sh"]

