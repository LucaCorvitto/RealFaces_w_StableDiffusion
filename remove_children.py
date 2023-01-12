import cv2
import os
import numpy as np
import shutil

# The model architecture
# download from: https://drive.google.com/open?id=1kiusFljZc9QfcIYdU2s7xrtWHTraHwmW
AGE_MODEL = 'weights/deploy_age.prototxt'
# The model pre-trained weights
# download from: https://drive.google.com/open?id=1kWv0AjxGSN0g31OeJa02eBGM0R_jcjIl
AGE_PROTO = 'weights/age_net.caffemodel'
# Each Caffe Model impose the shape of the input image also image preprocessing is required like mean
# substraction to eliminate the effect of illunination changes
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
# Represent the 8 age classes of this CNN probability layer
AGE_INTERVALS = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)',
                '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
# download from: https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
FACE_PROTO = "weights/deploy.prototxt.txt"
# download from: https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel
FACE_MODEL = "weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"
# Initialize frame size
frame_width = 512
frame_height = 512
# load face Caffe model
face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
# Load age prediction model
age_net = cv2.dnn.readNetFromCaffe(AGE_MODEL, AGE_PROTO)

def get_optimal_font_scale(text, width):
    """Determine the optimal font scale based on the hosting frame width"""
    for scale in reversed(range(0, 60, 1)):
        textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale/10, thickness=1)
        new_width = textSize[0][0]
        if (new_width <= width):
            return scale/10
    return 1

# from: https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]
    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image
    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))
    # resize the image
    return cv2.resize(image, dim, interpolation = inter)

def get_optimal_font_scale(text, width):
    """Determine the optimal font scale based on the hosting frame width"""
    for scale in reversed(range(0, 60, 1)):
        textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale/10, thickness=1)
        new_width = textSize[0][0]
        if (new_width <= width):
            return scale/10
    return 1

# from: https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]
    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image
    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))
    # resize the image
    return cv2.resize(image, dim, interpolation = inter)


def predict_age(input_path: str):
    """Predict the age of the faces showing in the image"""
    #print(input_path)
    # Read Input Image
    img = cv2.imread(input_path)
    # Take a copy of the initial image and resize it
    frame = img.copy()
    #if frame.shape[1] > frame_width:
        #frame = image_resize(frame, width=frame_width)
    #faces = get_faces(frame)
    #for i, (start_x, start_y, end_x, end_y) in enumerate(faces):
    face_img = frame
    # image --> Input image to preprocess before passing it through our dnn for classification.
    blob = cv2.dnn.blobFromImage(
        image=face_img, scalefactor=1.0, size=(227, 227), 
        mean=MODEL_MEAN_VALUES, swapRB=False
    )
    # Predict Age
    age_net.setInput(blob)
    age_preds = age_net.forward()
    #print("="*30, "Face Prediction Probabilities", "="*30)
    #for i in range(age_preds[0].shape[0]):
        #print(f"{AGE_INTERVALS[i]}: {age_preds[0, i]*100:.2f}%")
    i = age_preds[0].argmax()
    age = AGE_INTERVALS[i]
    age_confidence_score = age_preds[0][i]
    # Draw the box
    #label = f"Age:{age} - {age_confidence_score*100:.2f}%"
    #print(label)
    #print(age_preds[0])
    return age_preds[0]

def main():
    
    remove = []
    images = os.listdir('datasets/archive/')
    del_imgs = 0
    for input_p in images:
        input_path = 'datasets/archive/' + input_p
        age = predict_age(input_path)
        if sum(age[3::]) < 0.8:
	#if age[0] + age[1] + age[2] > 0.15 :
            remove.append(input_p)
            del_imgs += 1


    source_folder = "datasets/archive"
    destination_folder = "datasets/children"
    
    # iterate files
    for file in remove:
        # construct full file path
        source = os.path.join(source_folder, file)
        destination = os.path.join(destination_folder, file)
        # move file
        shutil.move(source, destination)

if __name__ == '__main__':
    main()
