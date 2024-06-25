# MediaPipe Body
import mediapipe as mp
import self
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import torch
from torchvision import transforms

import cv2
import threading
import time
import global_vars
import struct


from tensorflow.keras.models import load_model


class CaptureThread(threading.Thread):
    cap = None
    ret = None
    frame = None
    isRunning = False
    counter = 0
    timer = 0.0

    def run(self):
        self.cap = cv2.VideoCapture(1)  # sometimes it can take a while for certain video captures 4
        if global_vars.USE_CUSTOM_CAM_SETTINGS:
            self.cap.set(cv2.CAP_PROP_FPS, global_vars.FPS)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, global_vars.WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, global_vars.HEIGHT)

        time.sleep(1)

        while not global_vars.KILL_THREADS:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame
                self.isRunning = True
                if global_vars.DEBUG:
                    self.counter += 1
                    if time.time() - self.timer >= 3:
                        # print("Capture FPS: ", self.counter / (time.time() - self.timer))
                        self.counter = 0
                        self.timer = time.time()


# the body thread actually does the
# processing of the captured images, and communication with unity
class BodyThread(threading.Thread):
    data = ""
    dirty = True
    pipe = None
    timeSinceCheckedConnection = 0
    timeSincePostStatistics = 0


    # Load the gesture recognizer model


    def run(self):



        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose

        capture = CaptureThread()
        capture.start()

        with mp_pose.Pose(min_detection_confidence=0.80, min_tracking_confidence=0.5,
                          model_complexity=global_vars.MODEL_COMPLEXITY, static_image_mode=False,
                          enable_segmentation=True) as pose:

            while not global_vars.KILL_THREADS and capture.isRunning == False:
                print("Waiting for camera and capture thread.")
                time.sleep(0.5)
            print("Beginning capture")

            while not global_vars.KILL_THREADS and capture.cap.isOpened():
                ti = time.time()

                # Fetch stuff from the capture thread
                ret = capture.ret
                image = capture.frame

                # Image transformations and stuff
                # image = cv2.flip(image, 1)
                image.flags.writeable = global_vars.DEBUG

                # Detections
                results = pose.process(image)
                tf = time.time()





                # Rendering results
                if global_vars.DEBUG:
                    if time.time() - self.timeSincePostStatistics >= 1:
                       # print("Theoretical Maximum FPS: %f" % (1 / (tf - ti)))
                        self.timeSincePostStatistics = time.time()

                    if results.pose_landmarks:
                        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                  mp_drawing.DrawingSpec(color=(255, 100, 0), thickness=2,
                                                                         circle_radius=4),
                                                  mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2,
                                                                         circle_radius=2),
                                                  )
                    cv2.imshow('Body Tracking', image)
                    cv2.waitKey(1)

                if self.pipe == None and time.time() - self.timeSinceCheckedConnection >= 1:
                    try:
                        self.pipe = open(r'\\.\pipe\UnityMediaPipeBody', 'r+b', 0)
                        print("pipe created")
                    except FileNotFoundError:
                       # print("Waiting for Unity project to run...")
                        self.pipe = None
                    self.timeSinceCheckedConnection = time.time()

                if self.pipe != None:
                    # Set up data for piping
                    self.data = ""
                    i = 0

                    if results.pose_world_landmarks:
                        # image_landmarks = results.pose_landmarks
                        world_landmarks = results.pose_world_landmarks

                        # model_points = np.float32([[-l.x, -l.y, -l.z] for l in world_landmarks.landmark])
                        # image_points = np.float32([[l.x * image.shape[1], l.y * image.shape[0]] for l in image_landmarks.landmark])

                        #  body_world_landmarks_world = self.compute_real_world_landmarks(model_points,image_points,image.shape)
                        #   body_world_landmarks = results.pose_world_landmarks

                        for i in range(0, 33):
                            self.data += "FREE|{}|{}|{}|{}\n".format(i, results.pose_landmarks.landmark[i].x,
                                                                     results.pose_landmarks.landmark[i].y,
                                                                     results.pose_landmarks.landmark[i].z)
                        for i in range(0, 33):
                            self.data += "ANCHORED|{}|{}|{}|{}\n".format(i, world_landmarks.landmark[i].x,
                                                                         world_landmarks.landmark[i].y,
                                                                         world_landmarks.landmark[i].z)

                    s = self.data.encode('utf-8')
                    try:
                        self.pipe.write(struct.pack('I', len(s)) + s)
                        self.pipe.seek(0)
                        print("Data :", self.data)
                    except Exception as ex:
                        print("Failed to write to pipe. Is the unity project open?")
                        self.pipe = None

                # time.sleep(1/20)

        self.pipe.close()
        capture.cap.release()
        cv2.destroyAllWindows()




class GestureThread(threading.Thread):

        def run(self):
            mpHands = mp.solutions.hands
            hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
            mpDraw = mp.solutions.drawing_utils

            # Load the gesture recognizer model
            model = load_model('mp_hand_gesture')

            # Load class names
            f = open('gesture.names', 'r')
            classNames = f.read().split('\n')
            f.close()
            print(classNames)

            # Initialize the webcam
            cap = cv2.VideoCapture(1)

            # Initialize gesture_pipe
            gesture_pipe = None

            while True:
                # Read each frame from the webcam
                _, frame = cap.read()

                x, y, c = frame.shape

                # Flip the frame vertically
               # frame = cv2.flip(frame, 1)
                framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Get hand landmark prediction
                result = hands.process(framergb)

                # print(result)
                className = ''

                # post process the result
                if result.multi_hand_landmarks:
                    landmarks = []
                    for handslms in result.multi_hand_landmarks:
                        for lm in handslms.landmark:
                            # print(id, lm)
                            lmx = int(lm.x * x)
                            lmy = int(lm.y * y)

                            landmarks.append([lmx, lmy])

                        # Drawing landmarks on frames
                        mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

                        # Predict gesture
                        prediction = model.predict([landmarks])
                        # print(prediction)
                        classID = np.argmax(prediction)
                        className = classNames[classID]

                # show the prediction on the frame
                cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2, cv2.LINE_AA)

                print(className)
                # Show the final output
                cv2.imshow("Gesture Detection", frame)

                # Open named pipe and send gesture data
                if gesture_pipe is None:
                    try:
                        gesture_pipe = open(r'\\.\pipe\UnityMediaPipeGesture', 'r+b', 0)
                        print("pipe created")
                    except FileNotFoundError:
                        print("Waiting for Unity project to run...")
                        gesture_pipe = None

                if gesture_pipe is not None:
                    gesture_data = className
                    print(gesture_data)
                    gesture_data_encoded = gesture_data.encode('utf-8')
                    try:
                        gesture_pipe.write(struct.pack('I', len(gesture_data_encoded)) + gesture_data_encoded)
                        gesture_pipe.seek(0)
                        print("Gesture Data Sent:", gesture_data)
                    except Exception as ex:
                        print("Failed to write gesture data to pipe. Is the unity project open?")
                        gesture_pipe = None

                if cv2.waitKey(1) == ord('q'):
                    break

            # release the webcam and destroy all active windows
            cap.release()
            cv2.destroyAllWindows()

class OrganDetectionThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.loaded_model = torch.load(r'C:\Users\LEGION\Desktop\Bishnu\vit_model.pth')
        self.loaded_model.eval()
        self.class_names = ['Heart', 'Kidney']

    def run(self):
        cap = cv2.VideoCapture(1)
        organ_pipe= None;

        if not cap.isOpened():
            print("Error: Could not open webcam")
            return

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        while True:
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = transform(frame_rgb).unsqueeze(0)

                with torch.no_grad():
                    self.loaded_model.eval()
                    outputs = self.loaded_model(image)

                _, predicted = torch.max(outputs, 1)
                predicted_class = self.class_names[predicted.item()]

                cv2.putText(frame, f'Predicted Organ: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow('Organ Detection', frame)


                if organ_pipe is None:
                    try:
                        organ_pipe = open(r'\\.\pipe\UnityMediaPipeOrgan', 'r+b', 0)
                        print("Organ pipe created")
                    except FileNotFoundError:
                        print("Waiting for Unity project to run...")
                        organ_pipe = None

                if organ_pipe is not None:
                    organ_data = predicted_class
                    print(organ_data)
                    organ_data_encoded = organ_data.encode('utf-8')
                    try:
                        organ_pipe.write(struct.pack('I', len(organ_data_encoded)) + organ_data_encoded)
                        organ_pipe.seek(0)
                        print("Organ Data Sent:", organ_data)
                    except Exception as ex:
                        print("Failed to write organ data to pipe. Is the unity project open?")
                        organ_pipe = None

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("Error: Failed to capture frame")
                break

        cap.release()
        cv2.destroyAllWindows()


