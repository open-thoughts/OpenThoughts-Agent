FROM python:3.12.10-slim-bookworm
RUN pip install --no-cache-dir numpy==2.2.4 pandas==2.2.3 scipy==1.15.2 scikit-learn==1.6.1
ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
WORKDIR /app
COPY data /app/data
