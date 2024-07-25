# Lung Cancer Image Classification using Vision Transformers

This project implements a web application for classifying lung cancer images using Vision Transformers (ViT). The application allows users to upload an image of a lung cancer case and get the prediction. It also provides example cases for users to try.

## Overview

### Example Cases

The web application includes three example cases for demonstration purposes:

- **Bengin cases**
- **Malignant cases**
- **Normal cases**

### Vision Transformer Workflow

The Vision Transformer (ViT) model architecture is used for image classification. Below is a brief overview of the Vision Transformer workflow:

- **Patch Extraction:** Images are divided into smaller patches.
- **Position Embedding:** Positional information is added to each patch.
- **Transformer Encoder:** The patches are processed through multiple transformer layers.
- **Classification Head:** The final representation is used for classification.

## Application Screenshot

![Streamlit Web Application](https://github.com/user-attachments/assets/423fa1b7-8fb4-4988-95b0-a685065476f2)

### Example Images

<div style="display: flex; justify-content: space-around;">
    <div style="text-align: center;">
        <img src="https://github.com/user-attachments/assets/c657d535-8674-4713-a47c-ae35e237ff02" width="300"/>
        <p>Example 1: A sample lung cancer image.</p>
    </div>
    <div style="text-align: center;">
        <img src="https://github.com/user-attachments/assets/3e69bfb1-8f30-4c07-9b57-aa16dae884c9" width="300"/>
        <p>Example 2: A lung cancer image with patches overlaid.</p>
    </div>
</div>

![Transformer Workflow](https://github.com/user-attachments/assets/ebf0dd43-67c7-4f30-aaaa-cf7de06539d1)
*Vision Transformer Workflow: Patch extraction, position embedding, and transformer encoder layers.*

## Running the Application

### Prerequisites

- Python 3.8 or higher
- `pip` package manager

### Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/your-username/lung-cancer-classification.git
    cd lung-cancer-classification
    ```

2. Create and activate a virtual environment:

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

### Training the Model

1. Execute the Jupyter Notebook provided in the repository to train the model:

    ```sh
    jupyter notebook lung_cancer_classification.ipynb
    ```

2. After training, the model will be saved in the `vit-lung-cancer-model` directory with the following files:
    - `config.json`
    - `model.safetensors`
    - `preprocessor_config.json`

### Usage

1. Start the Streamlit application:

    ```sh
    streamlit run app.py
    ```
    
2. Open your web browser and navigate to `http://localhost:8501`.

3. Upload an image of a lung cancer case or use one of the provided example cases to see the classification result.


