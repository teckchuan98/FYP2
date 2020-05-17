import onnxruntime as ort
from Testing.utilities2 import detect, recognise, initialise, tag
import time
import cv2
from imutils import paths
import os
import sys



def main():
    ort_session = ort.InferenceSession('ultra_light_640.onnx')
    input_name = ort_session.get_inputs()[0].name
    ort_session, input_name, recognizer, le, (saved_embeds, names) = initialise()

    imagePaths = list(paths.list_images("val"))
    total = 0
    count = 0
    for (i, imagePath) in enumerate(imagePaths):
        total += 1
        name = imagePath.split(os.path.sep)[-2]
        frame = cv2.imread(imagePath)
        start = time.time()
        if frame is not None:
            _, temp = detect(frame, ort_session, input_name)
            face_locations, face_names, probability = recognise(temp, frame, recognizer, le, names, saved_embeds)
            frame = tag(frame, face_locations, face_names, probability)
            cv2.imwrite('output/' + str(i) + ".jpg", frame)

            #if len(face_names) > 0:
                #assert face_names[0] == name, "Test passed"
            name = imagePath.split(os.path.sep)[-1]
            x = ""

            for a in name:
                if not a.isdigit():
                    x = x + a
                else:
                    break
            if len(face_names) == 1:
                result = face_names[0]
                print(result, x)
                if result == x:
                    count += 1



        end = time.time()
        #print(end - start)
        #print(i)
    print(str(count) + " passed out of " + str(total))





if __name__ == "__main__":
    main()