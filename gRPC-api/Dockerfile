FROM python:3.8

RUN pip install -r requirements.txt

COPY . .

EXPOSE 50051

ENTRYPOINT ["python", "grpc_main.py"]