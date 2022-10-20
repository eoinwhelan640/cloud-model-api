import torch
# pillow lets us open image and do stuff to it
from PIL import Image
# to do inferences with pytorch already have, we need some torchvision & torch modules
from torchvision import transforms
from torchvision import models
from torch im port nn


# For an image to be analysed, it needs to go through transforms to work with the NN.
# standard transformations, eg normalise co-ords etc
def image_transformation(imagepath):
    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    image = Image.open(imagepath)
    imagetensor = test_transforms(image)
    return imagetensor


# catvdog.pt is a pretrained network made from kaggle data set with a bit of transfer learning.
# we're making a script that can interact with that model and give it a query image and allow
# it to predict

# This script is for an API so that the user can submit an image to this model in real time
# The model is already trained and we are just utilising the parameters
# run it on cpu, gpu isnt set up. Would use gpu if had available in real prod env
checkpoint = torch.load("catvdog.pt", map_location=torch.device("cpu"))
model = models.densenet121(pretrained=False)
model.classifier = nn.Sequential(nn.Linear(1024, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Dropout(0.1),
                                 nn.Linear(256, 2),
                                 nn.LogSoftmax(dim=1))

model.parameters = checkpoint['parameters']
# state_dict essentially loads in the weights
model.load_state_dict(checkpoint["state_dict"])
# Set to evaluation mode - basically makes it so that we can make an inference
model.eval()


if __name__ == "__main__":
    # Want to feed these to the model and use them to get the top prediction (top 1 prediction)
    graham_img_path = "../images/graham2.jpg"
    bronte_img_path = "../images/bronte.jpg"
    image_1 = image_transformation(graham_img_path)
    # need to make another transformation, adding a dimension because the signature of the model expects
    # these channels and we need to artificially make that happen
    image_1 = image_1[None, :, :, :]
    pred_1 = torch.exp(model(image_1))
    topconf_1, topclass_1 = pred_1.topk(1, dim=1)
    # print(topconf, topclass)
    image_2 = image_transformation(bronte_img_path)
    image_2 = image_2[None, :, :, :]
    pred_2 = torch.exp(model(image_2))
    topconf_2, topclass_2 = pred_2.topk(1, dim=1)
    # make sense of the prediction
    if topclass_1.item() == 1:
        print("Prediction 1 - ", {'class': 'dog', 'confidence': str(topconf_1.item())})
    else:
        print("Prediction 1 - ", {'class': 'cat', 'confidence': str(topconf_1.item())})
    if topclass_2.item() == 1:
        print("Prediction 2 - ", {'class': 'dog', 'confidence': str(topconf_2.item())})
    else:
        print("Prediction 2 - ", {'class': 'cat', 'confidence': str(topconf_2.item())})
