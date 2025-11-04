FROM continuumio/miniconda3:latest
WORKDIR /app
COPY environment.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src/predict.py .
COPY src/train.py .
COPY models ./models/final
COPY data_task1/evaluate/test.csv ./data_task1/evaluate/test.csv

CMD ["python", "predict.py"]