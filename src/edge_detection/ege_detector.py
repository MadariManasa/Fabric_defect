import cv2
import numpy as np


class CannyEdgeDetector:
    """
    Canny Edge Detection class for detecting edges in fabric images.
    Used for identifying fabric defects by detecting edge boundaries.
    """
    
    def __init__(self, threshold1=100, threshold2=200):
        """
        Initialize the Canny Edge Detector.
        
        Args:
            threshold1 (int): Lower threshold for the hysteresis procedure. Default: 100
            threshold2 (int): Upper threshold for the hysteresis procedure. Default: 200
        """
        self.threshold1 = threshold1
        self.threshold2 = threshold2
    
    def detect_edges(self, image):
        """
        Apply Canny edge detection to an image.
        
        Args:
            image (numpy.ndarray): Input image (grayscale or color)
            
        Returns:
            numpy.ndarray: Edge-detected image (binary)
        """
        # Convert to grayscale if the image is in color
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image
        
        # Apply Gaussian blur to reduce noise
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 1.5)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred_image, self.threshold1, self.threshold2)
        
        return edges
    
    def detect_edges_with_preprocessing(self, image, apply_morphology=True):
        """
        Apply Canny edge detection with optional morphological operations.
        
        Args:
            image (numpy.ndarray): Input image
            apply_morphology (bool): Whether to apply morphological operations. Default: True
            
        Returns:
            numpy.ndarray: Edge-detected image
        """
        # Detect edges using Canny
        edges = self.detect_edges(image)
        
        # Apply morphological operations to enhance edges
        if apply_morphology:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
            edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return edges
    
    def set_thresholds(self, threshold1, threshold2):
        """
        Update the Canny edge detection thresholds.
        
        Args:
            threshold1 (int): Lower threshold
            threshold2 (int): Upper threshold
        """
        self.threshold1 = threshold1
        self.threshold2 = threshold2
    
    def detect_edges_multi_scale(self, image, scales=[1, 0.5, 1.5]):
        """
        Apply Canny edge detection at multiple scales and combine results.
        
        Args:
            image (numpy.ndarray): Input image
            scales (list): List of scales to apply. Default: [1, 0.5, 1.5]
            
        Returns:
            numpy.ndarray: Combined edge-detected image
        """
        combined_edges = np.zeros(image.shape[:2], dtype=np.uint8)
        
        for scale in scales:
            # Resize image
            if scale != 1:
                h, w = image.shape[:2]
                resized = cv2.resize(image, (int(w * scale), int(h * scale)))
            else:
                resized = image
            
            # Detect edges
            edges = self.detect_edges(resized)
            
            # Resize back to original size if needed
            if scale != 1:
                edges = cv2.resize(edges, (image.shape[1], image.shape[0]))
            
            # Combine with existing edges
            combined_edges = cv2.bitwise_or(combined_edges, edges)
        
        return combined_edges


if __name__ == "__main__":
    # Example usage
    detector = CannyEdgeDetector(threshold1=100, threshold2=200)
    
    # Create a sample image for testing
    sample_image = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(sample_image, (150, 150), 50, (255, 255, 255), -1)
    cv2.rectangle(sample_image, (50, 50), (250, 250), (128, 128, 128), 2)
    
    # Detect edges
    edges = detector.detect_edges(sample_image)
    edges_with_morphology = detector.detect_edges_with_preprocessing(sample_image, apply_morphology=True)
    
    print("✓ Canny Edge Detection initialized successfully!")
    print(f"  - Original image shape: {sample_image.shape}")
    print(f"  - Detected edges shape: {edges.shape}")
    print(f"  - Edges with preprocessing shape: {edges_with_morphology.shape}")
    print(f"  - Threshold1: {detector.threshold1}, Threshold2: {detector.threshold2}")
    print("\n✓ Edge detection is ready to use!")
