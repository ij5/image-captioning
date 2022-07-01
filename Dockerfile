from python:3

RUN pip install transformers torch

COPY main.py /

CMD ["python", "/main.py"]