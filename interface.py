import os
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

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

        try:
            # Load the model
            model = load_model("model16.keras")
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return

        predicted_class, predicted_probability = predict_skin_disease(image)
        st.write(f'Predicted Class: {predicted_class}')
        # st.write(f'Probability: {predicted_probability:.2f}')

# Run the app
if __name__ == '__main__':
    main()
