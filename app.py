import sys
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
import traceback
import cv2
import mediapipe as mp
from preprocess_pipeline import FaceDetectionAlign
import time
from database import EmbeddingDatabase
from face_encoding import is_same_person
from anti_spoof_detection import FaceLivenessDetector


class MainWindow(QMainWindow):
    """Main window class for the Garment Inspection System"""
    
    def __init__(self):
        """Initialize the main window"""
        super().__init__()
        uic.loadUi("UI/main.ui", self)

        # models
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detector = self.mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

        self.face_recog = FaceDetectionAlign()
        self.face_liveness = FaceLivenessDetector()

        # import database class
        self.db = EmbeddingDatabase()

        try:
            # Initialize camera and set resolution left
            self.cap_up = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            self.cap_up.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap_up.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        except:
            traceback.print_exc()
            self.closeEvent()

        # flags
        self.last_frame = None
        self.process_face_recog = False
        self.face_detected_time = None

        self.add_new_face = False

        # Set up the timer for updating the camera feed
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(50)

        self.timer_2 = QTimer(self)
        self.timer_2.timeout.connect(self._face_recog)
        self.timer_2.start(2000)

        # add new button
        self.add.clicked.connect(lambda: setattr(self, 'add_new_face', True))


    def _face_recog(self):
        if self.process_face_recog is True:
            # stop the timer
            try:
                if self.last_frame is not None:

                    frame = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2RGB)
                    aligned_face, embedding = self.face_recog.process_image(frame)

                    annotated_img, is_real, score = self.face_liveness.detect(frame)

                    print("is_real: ", is_real)

                    if is_real is False:
                        # warn message
                        self.known_name.setText("Fake face detected!")
                        QMessageBox.warning(self, "Warning", "Fake face detected!")
                        return

                    if self.add_new_face is True:
                        self.add_new_face = False
                        text = self.name_text.toPlainText()

                        if text == "":
                            # warn message
                            QMessageBox.warning(self, "Warning", "Please enter a name!")
                            return
                        else:
                            # save the embedding to the database
                            if self.db.save_embedding(text, embedding):
                                QMessageBox.information(self, "Done", f"Face added successfully as {text}")
                                self.name_text.setPlainText("")
                            else:
                                QMessageBox.warning(self, "Input Error", "Name already exists!")
                                self.name_text.setPlainText("")

                    else:
                        all_knowns = self.db.list_all()

                        self.known_name.setText("")

                        # check for known faces
                        names = all_knowns.keys()

                        for name in names:
                            emb = all_knowns[name]
                            is_same, results = is_same_person(emb, embedding)
                            print("results: ",  results)
                            if is_same:
                                self.known_name.setText(name)
                                return None
                            
                        self.known_name.setText("Unknown")

                    self.process_face_recog = False
                    self.face_detected_time = None
            except:
                pass


    def update_frame(self):
        """Update the frame from the camera feed"""
        try:
            ret, frame = self.cap_up.read()
            if ret:
                # Convert the frame to RGB format
                self.last_frame  = frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (850, 480))

                results = self.face_detector.process(frame)

                if results.detections:
                    for detection in results.detections:
                        self.mp_drawing.draw_detection(frame, detection)

                        if self.face_detected_time is None:
                            self.face_detected_time = time.time()

                        # add 2 sec delay
                        if self.face_detected_time and time.time() - self.face_detected_time > 2:
                            self.process_face_recog = True
                        else:
                            self.process_face_recog = False

                else:
                    self.face_detected_time = None
                    self.process_face_recog = False

                # Convert the frame to QImage format
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                # Set the image to the label
                self.screen.setPixmap(QPixmap.fromImage(q_img))
        except:
            traceback.print_exc()
            self.closeEvent()


    def closeEvent(self, event):
        """Handle the close event of the main window"""
        try:
            # Release the camera and close the application
            self.cap_up.release()
            event.accept()
        except:
            traceback.print_exc()
            event.accept()


def main():
    """Main function to run the application"""
    app = QApplication(sys.argv)
    
    # Create and show the main window
    window = MainWindow()
    window.setWindowTitle("Face Unlock System")
    window.show()
    
    # Execute the application
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
