from python:3

RUN pip install transformers torch pillow flask

COPY main.py /

CMD ["python", "/main.py"]