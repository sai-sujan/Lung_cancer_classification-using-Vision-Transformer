import torch
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
import matplotlib.pyplot as plt

# Define constants
model_dir = './vit-lung-cancer-model'
image_path = 'bengin_image.jpg'  # Update this to the path of your image
num_labels = ['Bengin cases', 'Malignant cases', 'Normal cases']

# Load the model and feature extractor
model = ViTForImageClassification.from_pretrained(model_dir)
feature_extractor = ViTFeatureExtractor.from_pretrained(model_dir)

# Define a function to load and resize images
def load_and_resize(image_path):
    image = Image.open(image_path).convert("RGB")
    return image

# Load and preprocess the single image
image = load_and_resize(image_path)
inputs = feature_extractor(images=image, return_tensors='pt')

# Run inference
model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    preds = torch.argmax(outputs.logits, dim=-1)

# Print the prediction
predicted_label = num_labels[preds.item()]
print(f'The predicted label for the image is: {predicted_label}')

# # Display the image
# plt.imshow(image)
# plt.title(f'Predicted: {predicted_label}')
# plt.axis('off')
# plt.show()
