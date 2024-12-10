import os
import json
import requests
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
import base64
import io

# Set working directory and model path
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"

# Load the pre-trained model
model_plant = tf.keras.models.load_model(model_path)

# Loading the class names
class_indices = json.load(open(f"{working_dir}/class_indices.json"))

# API Key for PlantID
api_key = "HDpHFD0QtWtMSYVPPpJb37cYrxsGE01S3CfXAuG0D2DgzPyyxF"  # Replace with your actual PlantID API key


# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image, target_size=(224, 224)):
    img = image.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array


# Function to Predict the Class of an Image
def predict_image_class(model, image, class_indices):
    preprocessed_img = load_and_preprocess_image(image)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name


# Function to convert image to base64
def convert_image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


# Function to Fetch Disease Info from PlantID API
def fetch_disease_info_from_api(image):
    url = "https://api.plant.id/v3/identification"
    headers = {
        "Content-Type": "application/json",
        "Api-Key": api_key,
    }

    # Convert image to base64
    img_base64 = convert_image_to_base64(image)

    # Payload for the API request
    payload = {
        "images": [img_base64],
        "classification_level": "species",  # Use classification_level=species
        "health": "auto"  # Use health=auto for automatic health classification
    }

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 201:
        try:
            identification = response.json()
            disease_info = {
                'is_plant': identification['result']['is_plant']['binary'] == True,
                'suggestions': [],
            }

            # Extract suggestions
            for suggestion in identification['result']['classification']['suggestions']:
                name = suggestion.get('name', 'N/A')
                probability = suggestion.get("probability", 0)
                details = suggestion.get("details", {})  # Fetching details

                # Append the disease details
                disease_info['suggestions'].append({
                    "name": name,
                    "probability": probability,
                    "details": {
                        "description": details.get("description", "No description available."),
                        "treatment": details.get("treatment", "No treatment available."),
                        "wiki_link": details.get("wiki_link", "No wiki link available.")
                    },
                })
            return disease_info

        except ValueError as e:
            print("Error decoding JSON:", e)
            return None
    else:
        return None


# Streamlit App
st.set_page_config(page_title="Plant Disease Classifier", page_icon="🌱", layout="wide")

st.markdown('<h1 class="title">🌱 Plant Disease Classifier 🌱</h1>', unsafe_allow_html=True)
st.subheader("Upload an image of the plant to classify its disease")

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns([2, 3])

    with col1:
        resized_img = image.resize((200, 200))
        st.image(resized_img, caption="Uploaded Image", use_column_width=True)

    with col2:
        if st.button('Classify Disease', key='classify_button', help="Click to classify the disease"):
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)

            # Call the deep learning model to classify
            prediction = predict_image_class(model_plant, image, class_indices)
            st.success(f'**Prediction:** {str(prediction)}')

            st.markdown('</div>', unsafe_allow_html=True)

            # Fetch information using PlantID API and display
            st.markdown('<div class="report-box">', unsafe_allow_html=True)
            disease_info = fetch_disease_info_from_api(image)

            if disease_info:
                st.markdown(f"<h3 class='section-title'>Disease Information</h3>", unsafe_allow_html=True)
                st.markdown(f"**Is it a plant?** {'Yes' if disease_info['is_plant'] else 'No'}")
                for suggestion in disease_info['suggestions']:
                    disease_name = suggestion.get("name", "N/A")
                    probability = suggestion.get("probability", "N/A")
                    details = suggestion.get("details", {})

                    st.markdown(f"**Disease Name:** {disease_name}")
                    st.markdown(f"**Probability:** {probability:.2%}")
                    st.markdown(f"**Description:** {details.get('description', 'No description available.')}")
                    st.markdown(f"**Treatment:** {details.get('treatment', 'No treatment available.')}")
                    wiki_link = details.get('wiki_link')
                    if wiki_link:
                        st.markdown(f"[Learn More]({wiki_link})", unsafe_allow_html=True)


