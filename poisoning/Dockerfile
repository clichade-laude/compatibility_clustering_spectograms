FROM python:3.10.15

WORKDIR /home/app

RUN pip install numpy pillow

COPY poisoning/poison.py .

ENTRYPOINT ["python3", "poison.py"]

