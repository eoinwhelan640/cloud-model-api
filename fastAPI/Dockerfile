FROM python:3.8

RUN pip install -r requirements.txt

COPY . .
# UVICORNs default port to expose
EXPOSE 8000

# Associated command for the container - run our server
CMD ["uvicorn","fastapi_main:app", "--host","0.0.0.0","--port","8000"]
