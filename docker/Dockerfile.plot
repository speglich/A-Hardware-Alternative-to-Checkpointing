FROM python:3.10-slim AS devel

RUN pip3 --no-cache-dir install matplotlib pandas numpy tikzplotlib

WORKDIR /app
