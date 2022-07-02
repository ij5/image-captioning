from python:3

RUN pip install transformers torch pillow

COPY main.py /

CMD ["python", "/main.py"]