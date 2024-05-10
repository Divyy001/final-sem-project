import os
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Get the current directory
current_directory = os.path.dirname(__file__)

# Model file path
model_file = os.path.join(current_directory, "model2Vgg16added.keras")

# Check if the model file exists
if not os.path.isfile(model_file):
    st.error("Model file not found. Please check the file path.")
    st.stop()

# Load the model
model = load_model(model_file)

# Define class labels
class_labels = ['Seborrheic Keratoses and other Benign Tumors', 'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions',
                'Atopic Dermatitis Photos', 'Bullous Disease Photos', 'Cellulitis Impetigo and other Bacterial Infections',
                'Eczema Photos', 'Exanthems and Drug Eruptions', 'Hair Loss Photos Alopecia and other Hair Diseases',
                'Herpes HPV and other STDs Photos', 'Light Diseases and Disorders of Pigmentation',
                'Lupus and other Connective Tissue diseases', 'Melanoma Skin Cancer Nevi and Moles',
                'Nail Fungus and other Nail Disease', 'Poison Ivy Photos and other Contact Dermatitis',
                'Psoriasis pictures Lichen Planus and related diseases', 'Scabies Lyme Disease and other Infestations and Bites',
                'Acne and Rosacea Photos','Systemic Disease',
                'Tinea Ringworm Candidiasis and other Fungal Infections', 'Urticaria Hives', 'Vascular Tumors',
                'Vasculitis Photos', 'Warts Molluscum and other Viral Infections']

# Function to preprocess the image
def preprocess_image(image):
    resized_image = image.resize((256, 256))
    image_array = np.array(resized_image)
    normalized_image = image_array / 255.0
    preprocessed_image = np.expand_dims(normalized_image, axis=0)
    return preprocessed_image

# Function to make predictions
def predict_skin_disease(image):
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)[0]
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_labels[predicted_class_index]
    predicted_probability = predictions[predicted_class_index]
    return predicted_class, predicted_probability


# Streamlit app
def main():
    st.title('Skin Disease Classification')
    st.write('Upload an image of a skin lesion to classify the disease.')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        predicted_class, predicted_probability = predict_skin_disease(image)
        st.write(f'Predicted Class: {predicted_class}')
        # st.write(f'Probability: {predicted_probability:.2f}')


# Run the app
if __name__ == '__main__':
    main()
