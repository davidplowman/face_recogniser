#!/usr/bin/python3

from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QApplication, QLabel, QLineEdit, QGridLayout,
                             QHBoxLayout, QVBoxLayout, QWidget)
from typing import List, Dict, Tuple, Optional, Any, Union

import cv2
import numpy as np
import tensorflow as tf

from functools import partial

from picamera2 import CompletedRequest, MappedArray, Picamera2
from picamera2.job import Job
from picamera2.previews.qt import QGlPicamera2
from picamera2.devices import Hailo

from face_utils import SCRFDPostProc as PostProc

DETECTOR_NETWORK = "scrfd_2.5g_8l.hef"
RECOGNISER_NETWORK = "arcface_mobilefacenet_8l.hef"
REGISTER_FRAMES = 30  # Number of frames to capture when registering a face
MIN_SIZE = 64  # Minimum usable face size, in pixels.
SIMILARITY_THRESHOLD = 0.75  # Cosine similarity threshold for face comparisons
NUM_FACES = 6
INSTRUCTIONS = """
Enter name, stare straight at the camera and
press Enter to register. Wiggle your head
slightly for 2 seconds!

Delete name and press Enter to remove."""

class FaceApp(QWidget):
    """A PyQt-based face recognition application.

    This application provides a GUI for face detection and recognition using:
    - A face detection model (SCRFD) to locate faces in camera frames
    - A face recognition model (ArcFace) to generate face embeddings
    - A simple cosine similarity-based matching system

    The application allows users to:
    - Register new faces by providing a name and capturing multiple frames
    - Recognize registered faces in real-time
    - Remove registered faces
    """

    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    MAGENTA = (255, 0, 255)
    YELLOW = (0, 255, 255)
    CYAN = (255, 255, 0)
    # Reference points for face alignment:
    ARCFACE_POINTS = np.float32([[38.3, 51.7], [73.5, 51.5], [41.5, 92.4], [70.7, 92.2]])

    def __init__(self) -> None:
        """Initialize the face recognition application.

        This method:
        1. Sets up the face detection and recognition models
        2. Configures the camera for video capture
        3. Creates the GUI layout with registration controls
        4. Initializes state variables for face tracking
        """
        super().__init__()

        # Load the recogniser network, which generates the feature vector for a face.
        self.hailo_recog = Hailo(RECOGNISER_NETWORK)
        self.recog_model_h, self.recog_model_w, _ = self.hailo_recog.get_input_shape()
        print("Recognition model size", self.recog_model_w, "x", self.recog_model_h)

        # Load the detector network, which detects faces in the camera image.
        self.hailo_detect = Hailo(DETECTOR_NETWORK)
        self.detect_model_h, self.detect_model_w, _ = self.hailo_detect.get_input_shape()
        print("Detect model size", self.detect_model_w, "x", self.detect_model_h)
        anchors = {'steps': (8, 16, 32), 'min_sizes': ((16, 32), (64, 128), (256, 512))}
        self.detect_post_processor = PostProc(image_dims=(self.detect_model_h, self.detect_model_w),
                                                           anchors=anchors)
        self.video_w, self.video_h = self.detect_model_w, self.detect_model_h

        # Configure camera. Pass the "main" image to the network and the "lores" is for display.
        self.picam2 = Picamera2()
        main = {'size': (self.detect_model_w, self.detect_model_h), 'format': 'RGB888'}
        lores = {'size': (self.video_w, self.video_h), 'format': 'XRGB8888'}
        half_res = [d // 2 for d in self.picam2.sensor_resolution]
        sensor = {'output_size': half_res, 'bit_depth': 10}  # use 2x2 binned mode
        controls = {'FrameRate': 15}
        config = self.picam2.create_preview_configuration(main, lores=lores, sensor=sensor,
                                                          controls=controls, display='lores')
        self.picam2.configure(config)

        # Set up the GUI stuff.
        self.qpicamera2 = QGlPicamera2(self.picam2, width=self.video_w, height=self.video_h, keep_ar=False)
        self.qpicamera2.done_signal.connect(self.capture_done)
        layout_h = QHBoxLayout()
        layout_h.addWidget(self.qpicamera2, 70)
        layout_v = QVBoxLayout()
        self.registered_faces = []
        grid_layout = QGridLayout()
        for i in range(NUM_FACES):
            label = QLabel(f"Person {i}:")
            grid_layout.addWidget(label, i, 0)
            textbox = QLineEdit()
            textbox.returnPressed.connect(partial(self.enter_pressed, i))
            grid_layout.addWidget(textbox, i, 1)
            image = QLabel()
            image.setFixedSize(64, 64)
            grid_layout.addWidget(image, i, 2)
            self.registered_faces.append({"textbox": textbox, "name": "", "image": image, "vectors": []})
        layout_v.addLayout(grid_layout, 5)
        instructions = QLabel(INSTRUCTIONS)
        layout_v.addWidget(instructions)
        layout_h.addLayout(layout_v)
        self.setWindowTitle("Face Recogniser")
        self.resize(self.video_w + 320, self.video_h)
        self.setLayout(layout_h)

        self.registering = None
        self.register_frames = 0
        self.draw_rects = []
        self.draw_points = []
        self.draw_colour = self.RED
        self.draw_face = None
        self.draw_name = None

        # Finally start the camera.
        self.picam2.start()
        self.picam2.pre_callback = self.draw_callback
        self.picam2.capture_request(signal_function=self.qpicamera2.signal_done)

    def enter_pressed(self, i: int) -> None:
        """Handle the Enter key press in a registration textbox.

        This method is called when the user presses Enter in one of the registration textboxes.
        It either:
        - Starts the registration process for a new face if a name is provided
        - Removes a registered face if the name is empty

        Args:
            i (int): Index of the registration textbox that was activated
        """
        name = self.registered_faces[i]["textbox"].text()
        self.registered_faces[i]["name"] = name
        if name == "":
            self.registered_faces[i]["vectors"] = []
            self.registered_faces[i]["image"].clear()
        else:
            print(f"Registering {name}...")
            self.registered_faces[i]["textbox"].setEnabled(False)
            self.registering = i
            self.register_frames = REGISTER_FRAMES

    def draw_callback(self, request: CompletedRequest) -> None:
        """Draw annotations on the camera preview. Called automatically by the camera.

        This method is called before each frame is displayed and draws:
        - Bounding boxes around detected faces
        - Facial landmarks
        - The current face being analysed (if any)
        - The name of the recognized person (if any)

        Args:
            request (CompletedRequest): The camera request containing the frame to draw on
        """
        with MappedArray(request, 'lores') as m:
            for rect in self.draw_rects:
                cv2.rectangle(m.array, rect[:2], rect[2:], self.draw_colour, 2)
            for point in self.draw_points:
                cv2.circle(m.array, point, 3, self.draw_colour, -1)
            if self.draw_face is not None:
                h, w, _ = self.draw_face.shape
                m.array[:h, :w, :3] = self.draw_face
                m.array[h:h+1, :w, :] = 0
                m.array[:h, w:w+1, :] = 0
            if self.draw_name:
                (text_width, text_height), baseline = cv2.getTextSize(
                    self.draw_name, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                text_x = m.array.shape[1] - text_width - 5
                text_y = text_height + 5
                rectangle = m.array[text_y - text_height:text_y + baseline, text_x: text_x + text_width]
                rectangle[...] = rectangle // 2
                cv2.putText(m.array, self.draw_name, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    def capture_done(self, job: Job) -> None:
        """Process a captured camera frame. Called in response to a signal from capture_request().

        This method is called when a new frame is captured and:
        1. Runs face detection on the frame
        2. If a face is found:
           - Aligns and crops the face
           - Either registers it or tries to recognize it
           - Updates the display annotations
        3. Queues the next frame capture

        Args:
            job (Job): The completed camera capture job
        """
        # First get the image to pass to the detection network, and queue a request for the next frame.
        request = job.get_result()
        frame = request.make_array('main')
        request.release()
        self.picam2.capture_request(signal_function=self.qpicamera2.signal_done)

        # Run the detection network and post-processing, and unpack the results.
        results = self.hailo_detect.run(frame)
        post_proc_results = self.detect_post_processor.tf_postproc(results)
        detection_boxes = post_proc_results['detection_boxes'][0]
        num_detections = post_proc_results['num_detections'][0]
        face_landmarks = post_proc_results['face_landmarks'][0]
        n = int(num_detections.numpy())
        self.draw_rects = []
        self.draw_points = []
        self.draw_face = None
        self.draw_name = None

        if n:
            # Find the single largest face of any that were found, and the associated landmarks.
            def rect_area(r: List[float]) -> float:
                """Calculate the area of a bounding box."""
                x0, y0, x1, y1 = r
                return (x1 - x0) * (y1 - y0)
            areas = [rect_area(r) for r in detection_boxes[:n]]
            index = np.array(areas).argmax()
            rect = detection_boxes.numpy()[index].tolist()
            landmarks = face_landmarks.numpy()[index].reshape((-1, 2)).tolist()
            # Convert to image coordinates.
            rect = [max(0, int(v * s)) for v, s in zip(rect, (self.detect_model_w, self.detect_model_h) * 2)]
            landmarks = [[int(v * s) for v, s in zip(landmark, (self.detect_model_w, self.detect_model_h))]
                         for landmark in landmarks]

            # If the face was sufficiently large, crop and resize it for passing to the recogniser network.
            colour = self.RED
            face = None
            name = None
            if all([p1 > p0 + MIN_SIZE for p0, p1 in zip(rect[:2], rect[2:])]):
                # Get an aligned and cropped version of the face.
                face = self.create_crop(frame, rect, landmarks)
                raw_embeddings = self.hailo_recog.run(face)
                embeddings = tf.nn.l2_normalize(raw_embeddings).numpy()

                if self.registering is not None:
                    # Add this embedding vector to the set that we have registered for this face.
                    self.registered_faces[self.registering]["vectors"].append(embeddings)
                    self.register_frames -= 1
                    if self.register_frames == 0:
                        print("Registering done!")
                        self.registered_faces[self.registering]["textbox"].setEnabled(True)
                        small = cv2.resize(face, dsize=(64, 64))
                        img = QImage(small.data, small.shape[1], small.shape[0], QImage.Format_BGR888)
                        pixmap = QPixmap.fromImage(img)
                        self.registered_faces[self.registering]["image"].setPixmap(pixmap)
                        self.registering = None
                    # Wink the bounding box magenta/yellow/cyan while registering.
                    colour = (self.MAGENTA, self.YELLOW, self.CYAN)[(self.register_frames // 2) % 3]

                else:
                    # If any of the cosine similarities exceed the threshold, that will do.
                    embeddings = embeddings.T
                    scores = [sum([reg @ embeddings > SIMILARITY_THRESHOLD
                                   for reg in entry["vectors"]]) for entry in self.registered_faces]
                    best = np.argmax(scores)
                    score = scores[best]
                    name = self.registered_faces[best]["name"] if score > 0 else None
                    if name:
                        colour = self.GREEN

            # Update what we draw.
            self.draw_colour = colour
            self.draw_rects = [rect]
            self.draw_points = landmarks
            self.draw_face = face
            self.draw_name = name

    def create_crop(self, frame: np.ndarray, bbox: List[int], landmarks: List[List[int]]) -> np.ndarray:
        """Create an aligned and cropped face image.

        This method uses an affine transform to align the face based on eye and mouth positions,
        then crops it to the size expected by the recognition model.

        Args:
            frame (np.ndarray): The full camera frame
            bbox (List[int]): Bounding box coordinates [x1, y1, x2, y2]
            landmarks (List[List[int]]): Facial landmark coordinates

        Returns:
            np.ndarray: Aligned and cropped face image
        """
        l_eye, r_eye, nose, l_mouth, r_mouth = np.float32(landmarks)
        landmarks = [l_eye, r_eye, (l_mouth + r_mouth) / 2]
        l_eye, r_eye, l_mouth, r_mouth = self.ARCFACE_POINTS
        ref_points = [l_eye, r_eye, (l_mouth + r_mouth) / 2]
        M = cv2.getAffineTransform(np.float32(landmarks), np.float32(ref_points))
        return cv2.warpAffine(frame, M, (self.recog_model_w, self.recog_model_h))


if __name__ == "__main__":
    app = QApplication([])
    window = FaceApp()
    window.show()
    app.exec()
