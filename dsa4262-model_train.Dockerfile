FROM continuumio/miniconda3:latest

WORKDIR /app

COPY environment.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY src/preprocess.py .
COPY src/train.py .

RUN mkdir -p /data_task1
COPY data_task1/data.info.labelled.csv /data_task1/data.info.labelled.csv
COPY data_task1/dataset0.json /data_task1/dataset0.json

RUN mkdir -p /models/final
COPY models/best_params.json /models/best_params.json

CMD ["python", "train.py"]