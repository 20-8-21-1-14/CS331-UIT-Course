import cv2
import numpy as np

vid = cv2.VideoCapture(0)

while(True):
    ret, frame = vid.read()

    result = detector.detect_faces(frame)
    if len(result) > 0:

        bounding_box = result[0]['box']
        X = bounding_box[0]
        Y = bounding_box[1]
        W = bounding_box[2]
        H = bounding_box[3]
        cropped_image = frame[Y:Y+H, X:X+W]
        resize_cropped_img = cv2.resize(cropped_image, (224, 224))
        cv2.imwrite('frame.png', resize_cropped_img)

        search_vector = extract_vector(model, './frame.png')
        distance = np.linalg.norm(vectors - search_vector, axis=1)

        label = np.argmin(distance)
        text = ''
        if label == 0:
            text = 'Thuan'
        elif label == 1:
            text = 'Yen'

        cv2.rectangle(frame,
                      (bounding_box[0], bounding_box[1]),
                      (bounding_box[0]+bounding_box[2],
                       bounding_box[1] + bounding_box[3]),
                      (0, 155, 255),
                      2)

        cv2.putText(frame, text, (5, 30),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

        cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
