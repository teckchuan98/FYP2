import onnxruntime as ort
from Unit_Testing_Performance.utilities2 import detect, recognise, initialise, tag, remove_duplicate
import time
import cv2
from imutils import paths
import os
import sys



def main():
    ort_session = ort.InferenceSession('ultra_light_640.onnx')
    input_name = ort_session.get_inputs()[0].name
    ort_session, input_name, recognizer, le, (saved_embeds, names) = initialise()
    target_name = ["Donald Trump", "Tzuyu", "Tom Holland", "Aamir Khan"]

    imagePaths = list(paths.list_images("face_recognition_test_cases"))
    total = 0
    count = 0
    mis_clar = 0
    fail_clar = 0
    for (i, imagePath) in enumerate(imagePaths):
        name = imagePath.split(os.path.sep)[-2]
        frame = cv2.imread(imagePath)
        start = time.time()
        if frame is not None:
            _, temp = detect(frame, ort_session, input_name)
            face_locations, face_names, probability = recognise(temp, frame, recognizer, le, names, saved_embeds)
            face_names, probability, face_locations = remove_duplicate(face_names, face_locations, probability)
            frame = tag(frame, face_locations, face_names, probability)
            cv2.imwrite('face_recognition_test_output/' + str(i) + ".jpg", frame)
            name = imagePath.split(os.path.sep)[-1]
            x = ""

            for a in name:
                if not a.isdigit():
                    x = x + a
                else:
                    break

            if x not in target_name:
                if len(face_names) == 0:
                    count += 1
                else:
                    mis_clar += 1
            else:
                if len(face_names) == 1:
                    result = face_names[0]
                    if result == x:
                        count += 1
                    else:
                        mis_clar += 1
                elif len(face_names) > 1:
                    mis_clar += 1
                else:
                    fail_clar += 1
            total += 1


        end = time.time()
        #print(end - start)
        #print(i)
    print(str(count) + " correct result out of " + str(total))
    print(str(mis_clar) + " miss clarification  out of " + str(total))
    print(str(fail_clar) + " fail clarification  out of " + str(total))

if __name__ == "__main__":
    main()