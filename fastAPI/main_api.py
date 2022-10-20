# Design structure of the API, build a basic endpoint
from fastapi import FastAPI, HTTPException
import logging
from logging.config import dictConfig
from logger_config import log_config
from pydantic import BaseModel
from models import Models

# FastAPi does a lot for us. Allows us to define our endpoints, define function run when endpoint
# is called. All we're doing with "health" function is return that the service is online
app = FastAPI()
models = Models()
dictConfig(log_config)
logger = logging.getLogger("ml-ops")

# Neat feature of FastAPI making use of encapsulation.
# We can create the type ImagePayload, ie define it. This lets us declare inputs for functions to be
# this type and when that func gets an arg, python automatically picks it up as our created type.
# pydantic lib is being used for this. BaseModel is what we inherit
class ImagePayload(BaseModel):
    # we want our image coming in as a base64 image string. Check Swagger documentation
    # kind of like initialsing the type, same idea as casting empty string in kdb
    img_b64: str


# Health endpoint
@app.get("/health")
def health():
    #logging.info("Health request received.") - default at start
    logger.info("Health request received.") # our custom logger
    return "Service is online"

# To test endpoint, we need to run the webserver which is uvicorn.
# start up the virtualenv on cmd prompt and navigate to the fastAPI folder where main lives
# command is -"uvicorn main_api:app --reload"
# main_api:app is telling it where to look for the main application
# --reload is hot reload don't use reload on a prod server, it's debug mode essentially

# On our api we will have the option to classify using model built off tensorflow or pytorch
@app.post("/classify/tensorflow")
def tf_classify(request: ImagePayload):
    # using the defined imagePayload we created so that the request object (an image)
    # will automatically be deserialized into the created ImagePayload type

    # Good practice to wrap endpoints in a try except - notify caller about an error.
    try:
        logger.info("Tensorflow request received")
        # The request obj now ImagePayload type, so request.img_b64 works is set up, base64 image string
        img_array = models.tf_load_image(request.img_b64)
        result = models.tf_predict(img_array)
        return result
    except Exception as e:
        message = "Server error while processing image!"
        logger.error(f"{message}: {e}", exc_info=True) # exc_info outputs stack trace on server log
        raise HTTPException(status_code=500, detail=message) # message for user


@app.post("/classify/pytorch")
def pyt_classify(request: ImagePayload):
    try:
        logger.info("PyTorch request received")
        img_tensor = models.pyt_load_image(request.img_b64)
        result = models.pyt_predict(img_tensor)
        return result
    except Exception as e:
        message = "Server error while processing image!"
        logger.error(f"{message}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=message)