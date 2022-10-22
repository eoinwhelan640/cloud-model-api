import io
import os
import numpy as np
from PIL import Image

import tensorflow as tf
# some mix up with versions of keras/tensorflow and torch. Conflicts if not using exact set, so need to use version of keras where funcs are in image_utils
#from keras.utils.image_utils import load_img, img_to_array 
from keras.preprocessing.image import load_img,img_to_array

import torch
from torchvision import transforms, models
from torch import nn


class Models():
    def __init__(self) -> None:
        self.tf_model = tf.keras.models.load_model(os.path.join("..", "model_scripts","tf-params"))
        checkpoint = torch.load(os.path.join("..","model_scripts","catvdog.pt"),
                                map_location=torch.device("cpu"))
        self.pyt_model = models.densenet121(pretrained=False)
        self.pyt_model.classifier = nn.Sequential(nn.Linear(1024, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Dropout(0.1),
                                 nn.Linear(256, 2),
                                 nn.LogSoftmax(dim=1))

        self.pyt_model.parameters = checkpoint['parameters']
        self.pyt_model.load_state_dict(checkpoint["state_dict"])
        self.pyt_model.eval()

    # In our gRPC API, the interface can accept an image directly as a bytestream, so we dont
    # need to use the base64 for tensorflow or pytorch
    def tf_load_image(self, img_bytes: bytes) -> np.ndarray:
        img = Image.open(io.BytesIO(img_bytes))
        img = img.convert("RGB")
        img = img.resize((224,224), Image.NEAREST)
        img = img_to_array(img)
        img = img.reshape(1, 224, 224, 3)
        img = img.astype("float32")
        img = img - [123.68, 116.779, 103.939]
        return img

    def pyt_load_image(self, img_bytes: bytes) -> torch.Tensor:
        test_transforms = transforms.Compose([transforms.Resize(255),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])
        img = Image.open(io.BytesIO(img_bytes))
        image_tensor = test_transforms(img)
        image_tensor = image_tensor[None,:,:,:]
        return image_tensor

    def tf_predict(self, img_array: np.ndarray) -> dict:
        prediction = self.tf_model.predict(img_array)
        prediction_value = prediction[0][0]
        if prediction_value > 0.5:
            return {"class": "dog", "value": float(prediction_value)}
        else:
            return {"class": "cat", "value": float(prediction_value)}

    def pyt_predict(self, img_tensor: torch.Tensor) -> dict:
        prediction = torch.exp(self.pyt_model(img_tensor))
        topconf, topclass = prediction.topk(1, dim=1)

        if topclass.item() == 1:
            return {'class': 'dog', 'confidence': str(topconf.item())}
        else:
            return {'class': 'cat', 'confidence': str(topconf.item())}
