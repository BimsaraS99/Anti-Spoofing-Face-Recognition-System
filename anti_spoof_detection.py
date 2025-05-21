import cv2
import numpy as np
import os
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

class FaceLivenessDetector:
    def __init__(self, model_dir="./resources/anti_spoof_models", device_id=0):
        """
        Initialize the face liveness detector.
        
        Args:
            model_dir (str): Path to directory containing anti-spoofing models
            device_id (int): GPU device ID (0 for first GPU)
        """
        self.model_test = AntiSpoofPredict(device_id)
        self.image_cropper = CropImage()
        self.model_dir = model_dir
        
        # Verify model directory exists
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory '{model_dir}' not found!")
        
        # Verify there are models in the directory
        if not os.listdir(model_dir):
            raise ValueError(f"No models found in directory '{model_dir}'")

    def check_image_aspect_ratio(self, image):
        """
        Check if image has correct 3:4 aspect ratio required by the model.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            bool: True if aspect ratio is correct
        """
        height, width = image.shape[:2]
        return width / height == 3 / 4

    def adjust_aspect_ratio(self, image):
        """
        Adjust image to 3:4 aspect ratio by center cropping.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Cropped image
        """
        height, width = image.shape[:2]
        target_width = int(height * 3 / 4)
        start_x = (width - target_width) // 2
        return image[:, start_x:start_x + target_width]

    def detect(self, image):
        """
        Detect face liveness in an image.
        
        Args:
            image (numpy.ndarray): Input image (BGR format)
            
        Returns:
            tuple: (annotated_image, is_real, confidence_score)
                   - annotated_image: Image with detection results drawn
                   - is_real: Boolean indicating real (True) or fake (False)
                   - confidence_score: Confidence score between 0 and 1
        """
        # Create a copy of the original image for annotation
        annotated_image = image.copy()
        
        # Check and adjust aspect ratio if needed
        if not self.check_image_aspect_ratio(image):
            image = self.adjust_aspect_ratio(image)
            annotated_image = image.copy()
        
        # Get face bounding box
        image_bbox = self.model_test.get_bbox(image)
        
        if image_bbox is None:
            raise ValueError("No face detected in the image")
        
        prediction = np.zeros((1, 3))
        
        # Process with all models
        for model_name in os.listdir(self.model_dir):
            h_input, w_input, model_type, scale = parse_model_name(model_name)
            param = {
                "org_img": image,
                "bbox": image_bbox,
                "scale": scale,
                "out_w": w_input,
                "out_h": h_input,
                "crop": True,
            }
            if scale is None:
                param["crop"] = False
            img = self.image_cropper.crop(**param)
            prediction += self.model_test.predict(img, os.path.join(self.model_dir, model_name))
        
        # Get prediction result
        label = np.argmax(prediction)
        confidence_score = prediction[0][label] / 2
        is_real = label == 1
        
        # Draw results on image
        color = (0, 255, 0) if is_real else (0, 0, 255)
        label_text = "REAL" if is_real else "FAKE"
        
        cv2.rectangle(
            annotated_image,
            (image_bbox[0], image_bbox[1]),
            (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
            color, 2)
        
        cv2.putText(
            annotated_image,
            f"{label_text} (Score: {confidence_score:.2f})",
            (image_bbox[0], image_bbox[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2)
        
        return annotated_image, is_real, confidence_score

# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = FaceLivenessDetector()
    
    # Load test image
    image = cv2.imread("gov.jpg")
    
    try:
        # Detect liveness
        annotated_img, is_real, score = detector.detect(image)
        
        # Display results
        print(f"Result: {'Real' if is_real else 'Fake'}")
        print(f"Confidence Score: {score:.2f}")
        cv2.imshow("Liveness Detection", annotated_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Save result
        cv2.imwrite("result.jpg", annotated_img)
    except Exception as e:
        print(f"Error: {str(e)}")