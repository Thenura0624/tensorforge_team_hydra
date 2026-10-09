FROM python:3.14-slim
WORKDIR /service
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 JOB_DB_PATH=/service/runtime/jobs.sqlite3
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
ARG MODEL_KIND=classical
COPY requirements-neural.txt .
RUN if [ "$MODEL_KIND" = "neural" ]; then pip install --no-cache-dir torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r requirements-neural.txt; fi
COPY app ./app
COPY models ./models
COPY specs ./specs
COPY web ./web
RUN useradd --uid 10001 --create-home service && mkdir runtime && chown service:service runtime
USER service
VOLUME ["/service/runtime"]
EXPOSE 8000
ARG DEFAULT_MODEL_PATH=/service/models/baseline.joblib
ENV MODEL_PATH=$DEFAULT_MODEL_PATH
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
