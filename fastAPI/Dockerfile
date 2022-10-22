FROM python:3.8

COPY . .

RUN pip install -r requirements.txt
# UVICORNs default port to expose
EXPOSE 8000

# Associated command for the container - run our server
CMD ["uvicorn","fastapi_main:app", "--host","0.0.0.0","--port","8000"]
