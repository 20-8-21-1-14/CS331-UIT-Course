import cv2
import pickle
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
# from mtcnn import MTCNN

mtcnn = MTCNN()
resnet = InceptionResnetV1(pretrained='vggface2').eval()


def extract_feature(img_path):
    print("Extracting: ", img_path)
    img = cv2.imread(img_path)
    img_cropped = mtcnn(img)
    resnet.classify = True
    img_probs = resnet(img_cropped.unsqueeze(0))
    return img_probs.detach().numpy()


vectors = pickle.load(open("Face_Recogition_system\\face_vector.pkl", "rb"))

vid = cv2.VideoCapture(0)

while(True):
    ret, frame = vid.read()
    # print(frame.shape)
    if frame.any() == None:
        continue

    result, _, landmark = mtcnn.detect(frame, landmarks=True)
    x1, y1 = np.abs(result[0][0]), np.abs(result[0][1])
    x2, y2 = x1 + abs(result[0][2]), y1 + abs(result[0][3])
    print(x1, x2, y1, y2)
    if len(result) > 0:
        cropped_image = frame[int(y1):int(y2), int(x1):int(x2)]
        resize_cropped_img = cv2.resize(cropped_image, (224, 224))
        cv2.imwrite('frame.png', resize_cropped_img)

        search_vector = extract_feature('./frame.png')
        distance = np.linalg.norm(vectors - search_vector, axis=2)

        label = np.argmin(distance)
        text = ''
        if label == 0:
            text = 'Thuan'
        elif label == 1:
            text = 'Yen'

        cv2.rectangle(frame, (int(x1), int(y1)),
                      (int(x2), int(y2)), (0, 155, 255), 2)

        cv2.putText(frame, text, (5, 30),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

        cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
