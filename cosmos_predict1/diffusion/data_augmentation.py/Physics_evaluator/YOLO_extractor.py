# Import necessary libraries required for the YOLO model
import torch
import numpy as np
from ultralytics import YOLO
import torch.nn.functional as F
import os
from PIL import Image

# Setup YOLO model class, which shall take a model version input like Yolov8m or Yolo12 based on which the weights and everything will be loaded.
class YOLO_ADAE():
    def __init__(self, model_version='yolov12x-cls.pt', device=None, conf_threshold=0.25, iou_threshold=0.45):
        # Initialize the class with necessary parameters
        self.model_version = model_version
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = self.load_model(model_version)

    # Define a method to load the YOLO model
    def load_model(self, model_path):
        """
        Load the YOLO model with the specified model path or version.
        
        Args:
            model_path: Path to the model weights or model version string
            
        Returns:
            Loaded YOLO model
        """
        try:
            model = YOLO(model_path)
            model.to(self.device)
            print(f"Model loaded successfully on {self.device}")
            return model
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            return None

    # Define a method to run inference on the input 
    def run_inference(self, image_path, conf=None, iou=None):
        """
        Run inference on an input image.
        
        Args:
            image_path: Path to the input image or image array
            conf: Confidence threshold for detection (optional, uses default if not provided)
            iou: IoU threshold for non-maximum suppression (optional, uses default if not provided)
            
        Returns:
            Detection results
        """
        if self.model is None:
            print("Model not loaded properly.")
            return None
        
        conf = conf if conf is not None else self.conf_threshold
        iou = iou if iou is not None else self.iou_threshold
        
        try:
            results = self.model(image_path, conf=conf, iou=iou)
            return results
        except Exception as e:
            print(f"Error during inference: {e}")
            return None

    # Define a method to run feature extraction on the input by removing the last layer of detection/classification whatever
    def extract_features(self, image_path):
        """
        Extract features from the input image by removing the last detection/classification layer.
        
        Args:
            image_path: Path to the input image or image tensor
            
        Returns:
            Feature embeddings as numpy array
        """
        layer_indices = [10, 14, 17]  # Example layer indices, adjust as needed
        results = model.predict(source='path/to/your/video.mp4', embed=layer_indices)

        # Iterate over results to access the embeddings
        for result in results:
            embeddings = result.embeddings  # List of feature tensors from the specified layers
            # Now you can use these embeddings for your application
    
    # Define a method to run the cosine similarity on the two embedding which are return my feature extraction method
    def compute_similarity(self, embedding1, embedding2):
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding (numpy array)
            embedding2: Second embedding (numpy array)
            
        Returns:
            Cosine similarity score (float)
        """
        if embedding1 is None or embedding2 is None:
            print("One or both embeddings are None.")
            return None
        
        try:
            # Convert to tensors if they're numpy arrays
            if isinstance(embedding1, np.ndarray):
                embedding1 = torch.from_numpy(embedding1)
            if isinstance(embedding2, np.ndarray):
                embedding2 = torch.from_numpy(embedding2)
            
            # Ensure proper dimensions
            if embedding1.dim() == 1:
                embedding1 = embedding1.unsqueeze(0)
            if embedding2.dim() == 1:
                embedding2 = embedding2.unsqueeze(0)
                
            # Compute cosine similarity
            similarity = F.cosine_similarity(embedding1, embedding2).item()
            return similarity
            
        except Exception as e:
            print(f"Error computing similarity: {e}")
            return None