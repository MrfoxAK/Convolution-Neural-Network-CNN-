from PIL import Image
import cv2
from keras.models import load_model
import numpy as np

model = load_model("D:\\CU\\face_rec\\facefeatures_new_model.h5")

img_path = "D:\\CU\\face_rec\\Dataset\\g2.jpg"

img = cv2.imread(img_path)

face = cv2.resize(img,(224,224))

# face = cv2.flip(face,1)

# cv2.imwrite("g1.jpg",face)

im = Image.fromarray(face,'RGB')

im_array = np.array(im)

im_array = np.expand_dims(im_array, axis=0)
pred = model.predict(im_array)
print(pred)