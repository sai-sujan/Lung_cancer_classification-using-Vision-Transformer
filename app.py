import torch
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
import streamlit as st

# Define constants
model_dir = './vit-lung-cancer-model'
num_labels = ['Bengin cases', 'Malignant cases', 'Normal cases']

# Example image paths
example_images = {
    'Bengin cases': 'bengin_image.jpg',  # Update with your image path
    'Malignant cases': 'malignant_image.jpg',  # Update with your image path
    'Normal cases': 'normal_image.jpg'  # Update with your image path
}

# Load the model and feature extractor
model = ViTForImageClassification.from_pretrained(model_dir)
feature_extractor = ViTFeatureExtractor.from_pretrained(model_dir)


# Define a function to load and resize images
def load_and_resize(image_path):
    image = Image.open(image_path).convert("RGB")
    return image


# Define a function to predict the class of an image
def predict(image):
    inputs = feature_extractor(images=image, return_tensors='pt')
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=-1)
    return num_labels[preds.item()]


# Streamlit app
st.title("Lung Cancer Image Classification using Vision Transformers")
st.write("Upload an image of a lung cancer case and get the prediction.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = load_and_resize(uploaded_file)
    predicted_label = predict(image)

    # Print the prediction
    st.write(f'The predicted label for the image is: **{predicted_label}**')

    # Display the image with the prediction
    st.image(image, caption=f'Predicted: {predicted_label}', use_column_width=True)

# Display example images
st.write("## Example Cases")
cols = st.columns(3)
for i, (label, image_path) in enumerate(example_images.items()):
    with cols[i]:
        st.image(image_path, caption=label, use_column_width=True)
        if st.button(f"Predict for {label}"):
            image = load_and_resize(image_path)
            predicted_label = predict(image)
            st.write(f'The predicted label for this image is: **{predicted_label}**')
