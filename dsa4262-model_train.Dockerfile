FROM continuumio/miniconda3:latest

WORKDIR /app

COPY environment.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY src/preprocess.py .
COPY src/train.py .
CMD ["python", "train.py"]