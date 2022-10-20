#from numpy.lib.npyio import load
import tensorflow as tf
from keras.utils import load_img, img_to_array
import os


def load_image(filename):
    # Write a function that loads the image and convert image to array
    img = load_img(filename, target_size=(224, 224))
    img = img_to_array(img)
    # then reshape image to a single sample with 3 channels
    img = img.reshape(1, 224, 224, 3)
    # center pixel data normalize the co-ordinates
    img = img.astype("float32")
    img = img - [123.68, 116.779, 103.939]
    return img


# load the model - whole folder
model = tf.keras.models.load_model("tf-params")


if __name__ == "__main__":
    img_1 = load_image(os.path.join("..", "images", "paisley.jpg"))
    img_2 = load_image(os.path.join("..", "images", "graham2.jpg"))

    pred_1 = model.predict(img_1)
    pred_2 = model.predict(img_2)

    if pred_1[0, 0] > 0.5:
        print(f"Prediction 1 {pred_1[0,0]:.2f} - dog")
    else:
        print(f"Prediction 1 {pred_1[0,0]:.2f} - cat")

    if pred_2[0, 0] > 0.5:
        print(f"Prediction 2 {pred_2[0,0]:.2f} - dog")
    else:
        print(f"Prediction 2 {pred_2[0,0]:.2f} - cat")
