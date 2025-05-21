import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet


class FaceDetectionAlign:
    """
        Class responsible for:
        - Detecting faces in an image,
        - Performing face alignment based on detected landmarks,
        - Returning a standardized face image aligned and cropped to a consistent size,
        ready for face recognition or embedding extraction.
    """

    def __init__(self, output_size=160):
        self.detector = MTCNN()

        self.model = FaceNet()

        template_112 = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]
        ], dtype=np.float32)

        scale = output_size / 112.0
        self.dst_landmarks = template_112 * scale

        self.output_size_img = (output_size, output_size)


    def __detect_faces(self, image):
        faces = self.detector.detect_faces(image)
        if not faces:
            return None
        # Sort faces by confidence score
        faces = sorted(faces, key=lambda x: x['confidence'], reverse=True)
        return faces


    def __get_main_face(self, detections):
        """
        Takes a list of face detections and returns the one with the largest area (assumed to be closest).
        """
        if not detections:
            return None

        # Compute area for each face and find the one with the largest bounding box
        largest_face = max(detections, key=lambda d: d['box'][2] * d['box'][3])
        
        # Optional: clean up NumPy types to native Python types
        largest_face_cleaned = {
            'box': list(map(int, largest_face['box'])),
            'confidence': float(largest_face['confidence']),
            'keypoints': {k: list(map(int, v)) for k, v in largest_face['keypoints'].items()}
        }
        return largest_face_cleaned


    def __align_face(self, image, bbox, landmarks):
        # Unpack bbox
        x1, y1, w, h = bbox
        x2, y2 = x1 + w, y1 + h

        # Ensure coordinates are within image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)

        # Crop face from the image
        face = image[y1:y2, x1:x2]

        # Shift landmarks relative to the cropped region
        src_landmarks = np.array([
            [landmarks['left_eye'][0] - x1, landmarks['left_eye'][1] - y1],
            [landmarks['right_eye'][0] - x1, landmarks['right_eye'][1] - y1],
            [landmarks['nose'][0] - x1, landmarks['nose'][1] - y1],
            [landmarks['mouth_left'][0] - x1, landmarks['mouth_left'][1] - y1],
            [landmarks['mouth_right'][0] - x1, landmarks['mouth_right'][1] - y1]
        ], dtype=np.float32)


        # Estimate affine transform matrix
        transform_matrix, _ = cv2.estimateAffinePartial2D(src_landmarks, self.dst_landmarks, method=cv2.LMEDS)

        # Apply affine transform to crop and align face
        aligned_face = cv2.warpAffine(face, transform_matrix, self.output_size_img, borderValue=0.0)

        return aligned_face
    

    def encode(self, image):
        """
        Encodes the given image to a 128D vector.
        :param image: The image to encode.
        :return: The encoded image.
        """
        return self.model.embeddings(image)


    def process_image(self, image):
        
        """
        Process the input image to detect and align faces.
        """
        # scale image to 1200 width
        scale = 1200 / image.shape[1]
        image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)), interpolation=cv2.INTER_LINEAR)
        
        # Detect faces
        faces = self.__detect_faces(image)

        main_face = self.__get_main_face(faces) 

        aligned_face = self.__align_face(image, main_face['box'], main_face['keypoints']) if main_face else None
        
        aligned_face_facenet = np.expand_dims(aligned_face, axis=0)

        # Encode the aligned face
        embedding = self.encode(aligned_face_facenet)

        return aligned_face, embedding
    


        