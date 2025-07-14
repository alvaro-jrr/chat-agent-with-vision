import os
from typing import Union

from keras.models import Model, load_model
from keras.utils import img_to_array, load_img
import numpy as np

class Vision:
  """
  The class to handle the vision.
  """

  # The model.
  model: Model

  # The image size.
  IMAGE_SIZE = (64, 64)

  def __init__(self):
    self.model = load_model("model.keras")

  def load_image(self, image_path: str) -> np.ndarray:
    """Load the image and preprocess it for the model."""
    image_path = image_path.strip().strip("'")

    # Check if the image exists.
    if not os.path.exists(image_path):
      raise FileNotFoundError(f"Image not found: {image_path}")

    image = load_img(image_path, target_size=self.IMAGE_SIZE)
    image = img_to_array(image)
    image = np.array(image) / 255.0

    return image

  def predict(self, image: Union[np.ndarray, str]) -> tuple:
    """Predict the class of the image."""

    if isinstance(image, str):
      image = self.load_image(image)
      
    # Predict the class.
    prediction = self.model.predict(np.array([image]), verbose=0)
    
    developer_prediction = np.argmax(prediction[0])
    expression_prediction = np.argmax(prediction[1])

    return developer_prediction, expression_prediction

  def get_developer_name(self, prediction: int) -> str:
    """Get the name of the developer."""
    
    return ['aguinagalde', 'resplandor'][prediction]

  def get_expression_name(self, prediction: int) -> str:
    """Get the name of the expression."""
    
    return ['angry', 'happy', 'laughing', 'sad', 'surprised', 'thoughtful', 'tired'][prediction]