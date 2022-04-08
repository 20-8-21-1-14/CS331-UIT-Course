import cv2
from unittest import result
from mtcnn import MTCNN
import cv2
'''
# Take image from webcam
cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:
        # SPACE pressed
        img_name = "person_{}.jpg".format(img_counter)
        cv2.imwrite('Face_Recogition_system\\org_images\\' + img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()

'''

# crop face from taken image
img = cv2.cvtColor(cv2.imread(
    "Face_Recogition_system\\org_images\\person_1.jpg"), cv2.COLOR_BGR2RGB)
detector = MTCNN()
result = detector.detect_faces(img)


bounding_box = result[0]['box']
X = bounding_box[0]
Y = bounding_box[1]
W = bounding_box[2]
H = bounding_box[3]
cropped_image = img[Y:Y+H, X:X+W]
resize_cropped_img = cv2.resize(cropped_image, (224, 224))
cv2.imwrite('Face_Recogition_system\\faces\\2.png',
            resize_cropped_img[:, :, ::-1])

cv2.waitKey()
cv2.destroyAllWindows()
