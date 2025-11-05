FROM continuumio/miniconda3:latest
WORKDIR /app
COPY environment.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src/preprocess.py .
COPY src/predict.py .
COPY src/train.py .

RUN mkdir -p /models/final
COPY models/final /models/final

RUN mkdir -p /data_task1/evaluate
COPY data_task1/evaluate/test.csv /data_task1/evaluate/test.csv

CMD ["python", "predict.py"]