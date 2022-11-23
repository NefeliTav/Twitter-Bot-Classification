FROM python:3.7-stretch

RUN apt-get update
RUN apt-get install default-jdk -y

COPY . .

RUN pip install -r requirements.txt

CMD python app.py