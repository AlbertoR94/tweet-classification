FROM python:3.9.12-slim

ENV PYTHONUNBUFFERED = TRUE

WORKDIR /app

COPY ["requirements.txt", "Makefile", "processed_train.csv", "./"]
RUN pip install --upgrade pip &&\
	pip install --no-cache-dir -r requirements.txt

COPY ["*.py", "model.joblib", "pipeline.joblib", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "app:app"]