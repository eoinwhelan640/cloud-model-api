FROM python:3.8

COPY . .

RUN pip install -r requirements.txt

EXPOSE 50051

ENTRYPOINT ["python", "grpc_main.py"]