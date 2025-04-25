import os
import json
import requests
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
import base64
import io
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import gdown


st.set_page_config(page_title="Plant Disease Diagnosis", page_icon="🌿", layout="wide")


working_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = f"{working_dir}/trained_model"
model_path = f"{model_dir}/plant_disease_prediction_model.h5"

# Create the model directory if it doesn't exist
os.makedirs(model_dir, exist_ok=True)

# Check if the model exists
if not os.path.exists(model_path):
    # Google Drive file ID extracted from your link
    file_id = "1rKh-IElSdHTqax7XdfSdZTn-r8T_qWPf"
    
    # Construct the download URL
    url = f"https://drive.google.com/uc?id={file_id}"
    
    # Show download status in Streamlit
    st.write("Downloading model from Google Drive...")
    
    try:
        # Download the file from Google Drive
        gdown.download(url, model_path, quiet=False)
        st.write("Model downloaded successfully!")
    except Exception as e:
        st.error(f"Error downloading model: {e}")
        st.stop()
else:
    st.write("Model file already exists, using cached version.")

# Load the pre-trained model
try:
    model_plant = tf.keras.models.load_model(model_path)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Loading the class names
class_indices = json.load(open(f"{working_dir}/class_indices.json"))

# API Key for PlantID (securely stored)
plant_id_api_key = "T1BqvfuHK6DjZBewan7issk6BLm8nnX5iu5KtE7SzNPbMwPo94"  # Use environment variables for security

# Gemini API Key
gemini_api_key = "AIzaSyA0_HPs7aKEP8w-J3S3YS1858ffsEMK22A"  # Better to use st.secrets or environment variables

# Initialize Google Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    google_api_key=gemini_api_key,
    temperature=0.2,
    max_output_tokens=1024,
)

# Create a prompt template for disease information
disease_prompt = ChatPromptTemplate.from_template("""
You are a plant pathology expert. Provide detailed information about the following plant disease:
{disease_name}

Include the following information:
1. Brief description of the disease
2. Common symptoms
3. Causes of the disease
4. Plants commonly affected
5. Prevention and treatment methods

Keep your response concise and formatted in markdown.
""")

# Create a prompt template specifically for treatment information
treatment_prompt = ChatPromptTemplate.from_template("""
You are a plant disease treatment specialist. For the plant disease "{disease_name}", provide ONLY prevention and treatment methods.

Format your response as a clear, concise list of treatment methods. Include:
- Cultural practices
- Chemical treatments (if applicable)
- Biological controls (if applicable)
- Prevention strategies

Keep your response under 300 words and use markdown formatting.
""")

# Create a prompt template for the chatbot
chatbot_prompt = ChatPromptTemplate.from_template("""
You are PlantMedic, a knowledgeable plant disease expert chatbot. You provide helpful advice about plant diseases, gardening, and plant care.

The user is asking the following question:
{user_query}

If they're asking about a specific plant disease, provide accurate information. If they ask about plant care in general, give helpful advice.
If their question is unrelated to plants, gardening, or plant diseases, politely guide them back to plant-related topics.

Keep your response concise, friendly, and formatted in markdown.
""")

# Create the chains
disease_info_chain = disease_prompt | llm
treatment_info_chain = treatment_prompt | llm
chatbot_chain = chatbot_prompt | llm


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
    predicted_class_name = class_indices.get(str(predicted_class_index), "Unknown Class")
    return predicted_class_name, predictions[0][predicted_class_index]


# Function to convert image to base64
def convert_image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def fetch_disease_info_from_api(image):
    url = "https://api.plant.id/v3/identification"
    headers = {
        "Content-Type": "application/json",
        "Api-Key": plant_id_api_key,
    }

    img_base64 = convert_image_to_base64(image)
    payload = {
        "images": [img_base64],
        "classification_level": "species",
        "health": "auto"
    }

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 201:
        try:
            identification = response.json()
            disease_info = {
                'model_version': identification.get('model_version', 'Unknown'),
                'suggestions': [],
            }

            for suggestion in identification['result']['disease']['suggestions']:
                name = suggestion.get('name', 'N/A')
                probability = suggestion.get('probability', 0)
                probability_percentage = probability * 100

                if name.lower() != "fungi":
                    disease_info['suggestions'].append({
                        "name": name,
                        "probability": probability_percentage,
                    })
            return disease_info

        except (ValueError, KeyError) as e:
            print("Error decoding JSON or extracting details:", e)
            return None
    else:
        print(f"API request failed with status code {response.status_code}: {response.text}")
        return None


# Function to get detailed disease information using Gemini LLM
def get_disease_details(disease_name):
    try:
        # Invoke the chain with the disease name
        response = disease_info_chain.invoke({"disease_name": disease_name})
        return response.content
    except Exception as e:
        print(f"Error getting disease details: {e}")
        return f"Could not fetch detailed information for {disease_name}. Please try again."


# Function to get only treatment information using Gemini LLM
def get_treatment_details(disease_name):
    try:
        # Invoke the treatment chain with the disease name
        response = treatment_info_chain.invoke({"disease_name": disease_name})
        return response.content
    except Exception as e:
        print(f"Error getting treatment details: {e}")
        return f"Could not fetch treatment information for {disease_name}. Please try again."


# Function to get chatbot response using Gemini
def get_chatbot_response(user_query):
    try:
        # Invoke the chatbot chain with the user query
        response = chatbot_chain.invoke({"user_query": user_query})
        return response.content
    except Exception as e:
        print(f"Error getting chatbot response: {e}")
        return "I'm having trouble processing your question. Please try again."


# Custom CSS with beautiful color scheme


st.markdown("""
    <style>
    :root {
        --primary-color: #2E7D32;
        --primary-light: #4CAF50;
        --primary-dark: #1B5E20;
        --accent-color: #FF9800;
        --accent-light: #FFCC80;
        --bg-color: #F9FBE7;
        --card-bg: #FFFFFF;
        --text-color: #263238;
        --text-light: #546E7A;
        --success-color: #00897B;
        --warning-color: #FF8F00;
        --danger-color: #D32F2F;
    }

    .main {
        padding: 2rem;
        max-width: 1200px;
        margin: 0 auto;
        background-color: var(--bg-color);
    }

    .stApp {
        background-color: var(--bg-color);
    }

    h1 {
        color: white;
        text-align: center;
        padding: 1.5rem;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, var(--primary-light) 0%, var(--primary-dark) 100%);
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }

    h2, h3, h4 {
        color: var(--primary-dark);
    }

    .section {
        background: var(--card-bg);
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        margin-bottom: 2rem;
        border: 1px solid #E0F2F1;
    }

    .upload-section {
        background: linear-gradient(to right, #E8F5E9, #F1F8E9);
        border-left: 5px solid var(--primary-color);
        transition: all 0.3s ease;
    }

    .upload-section:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.1);
    }

    .stButton > button {
        background: linear-gradient(45deg, var(--primary-color), var(--primary-light));
        color: white;
        padding: 0.7rem 2.5rem;
        border-radius: 30px;
        border: none;
        font-weight: 600;
        box-shadow: 0 4px 10px rgba(46, 125, 50, 0.3);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stButton > button:hover {
        background: linear-gradient(45deg, var(--primary-dark), var(--primary-color));
        box-shadow: 0 6px 15px rgba(46, 125, 50, 0.4);
        transform: translateY(-2px);
    }

    .disease-card {
        background: #FAFAFA;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
        border-top: 5px solid var(--primary-light);
    }

    .disease-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
    }

    .confidence-high {
        border-top-color: var(--success-color);
    }

    .confidence-medium {
        border-top-color: var(--warning-color);
    }

    .confidence-low {
        border-top-color: var(--danger-color);
    }

    .disease-details {
        background: #F1F8E9;
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
        border: 1px solid #DCEDC8;
        box-shadow: inset 0 1px 5px rgba(0, 0, 0, 0.05);
    }

    .treatment-details {
        background: linear-gradient(to right bottom, #E8F5E9, #F1F8E9);
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
        border-left: 4px solid var(--success-color);
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.05);
    }

    .header-section {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
    }

    .header-section svg {
        margin-right: 10px;
    }

    .footer {
        text-align: center;
        padding: 1.5rem;
        color: var(--text-light);
        font-size: 0.9rem;
        margin-top: 2rem;
        background: var(--card-bg);
        border-radius: 10px;
        box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.03);
    }

    /* Progress bar styles */
    .progress-container {
        width: 100%;
        height: 10px;
        background-color: #E0E0E0;
        border-radius: 5px;
        margin-top: 10px;
        overflow: hidden;
    }

    .progress {
        height: 100%;
        border-radius: 5px;
        transition: width 0.5s ease;
    }

    /* Tab styles */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background-color: #F5F5F5;
        padding: 0.5rem;
        border-radius: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #FAFAFA;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        color: var(--text-light);
        border: 1px solid #E0E0E0;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, var(--primary-light), var(--primary-color));
        color: white !important;
        border: none;
        font-weight: 600;
    }

    /* Image container */
    .image-container {
        padding: 1rem;
        background: white;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }

    .image-container:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
    }

    /* File uploader */
    .stFileUploader > div > label {
        color: var(--primary-dark);
        font-weight: 500;
    }

    .stFileUploader > div > div > button {
        background-color: var(--primary-light);
    }

    /* Chatbot styles */
    .chat-container {
        border-radius: 12px;
        background: var(--card-bg);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        padding: 1rem;
        margin-bottom: 1rem;
        max-height: 400px;
        overflow-y: auto;
        display: flex;
        flex-direction: column;
    }

    .chat-message {
        margin-bottom: 1rem;
        padding: 1rem;
        border-radius: 10px;
        max-width: 80%;
    }

    .user-message {
        align-self: flex-end;
        background: linear-gradient(45deg, var(--primary-light), var(--primary-color));
        color: white;
        border-bottom-right-radius: 0;
    }

    .bot-message {
        align-self: flex-start;
        background: #F5F5F5;
        border: 1px solid #E0E0E0;
        border-bottom-left-radius: 0;
    }

    .chat-input {
        display: flex;
        margin-top: 1rem;
    }

    .chat-input input {
        flex-grow: 1;
        padding: 0.75rem;
        border-radius: 30px;
        border: 1px solid #E0E0E0;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
    }

    .chat-tab {
        background: linear-gradient(to right, #E8F5E9, #F1F8E9);
        border-radius: 10px;
        padding: 1rem;
        border-left: 5px solid var(--primary-light);
    }

    /* Gemini branding */
    .gemini-powered {
        display: inline-block;
        font-size: 0.8rem;
        background: linear-gradient(120deg, #4285F4, #EA4335, #FBBC05, #34A853);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 0.2rem 0.5rem;
        margin-left: 0.5rem;
        border-radius: 4px;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# App Header with Icon
# App Header with Icon
image_path = os.path.join(working_dir, "stem_5776393.png")

# App Header with Icon (Better solution)
if os.path.exists(image_path):
    # Create a two-column layout for the title
    title_col1, title_col2 = st.columns([1, 10])
    with title_col1:
        # Display the image
        plant_icon = Image.open(image_path)
        st.image(plant_icon, width=80)
    with title_col2:
        # Display the title
        st.markdown("""
        <h1 style="margin-top:0;">Plant Disease Classifier & Treatment</h1>
        """, unsafe_allow_html=True)
else:
    # Fallback if image is not found
    st.markdown("""
    <h1>
        <svg xmlns="http://www.w3.org/2000/svg" width="50" height="50" fill="currentColor" viewBox="0 0 16 16" style="vertical-align: middle; margin-right: 10px;">
            <path d="M1 3.5c0 .9.6 1.6 1.4 1.9-.1.6-.2 1.1-.2 1.8 0 4.1 3.4 7.5 7.5 7.5s7.5-3.4 7.5-7.5c0-.6-.1-1.2-.2-1.8.8-.3 1.4-1 1.4-1.9 0-1.1-.9-2-2-2-.7 0-1.4.4-1.7 1-.9-.4-1.9-.7-3-.7-1 0-2 .2-2.9.7-.3-.6-1-1-1.7-1C2 1.5 1 2.4 1 3.5z"/>
            <path d="M8 4.5c.8 0 1.5.3 2.1.7.3-.5.9-.8 1.4-.8 1 0 1.8.8 1.8 1.8 0 .8-.5 1.5-1.3 1.7.1.5.2 1 .2 1.5 0 2.9-2.4 5.3-5.3 5.3S1.8 11.3 1.8 8.4c0-.5.1-1 .2-1.5-.7-.2-1.3-.9-1.3-1.7 0-1 .8-1.8 1.8-1.8.6 0 1.1.3 1.4.8.6-.4 1.3-.7 2.1-.7z"/>
        </svg>
        Plant Disease Classifier & Treatment
    </h1>
    """, unsafe_allow_html=True)

# App Description
with st.container():
    st.markdown('<div class="section">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown("""
        ### 🌱 Identify and Treat Plant Diseases

        This application helps diagnose plant diseases from images and provides effective treatment methods:

        1. **Upload** a photo of your plant showing symptoms
        2. **Analyze** to identify potential diseases
        3. **Get Treatments** specifically for your plant's condition
        4. **Chat with PlantMedic** for personalized plant care advice
        """)
    st.markdown('</div>', unsafe_allow_html=True)

# Main App Navigation
app_mode = st.sidebar.selectbox("Choose App Mode", ["Disease Diagnosis", "Plant Care Chatbot"])

if app_mode == "Disease Diagnosis":
    # Upload Section
    with st.container():
        st.markdown('<div class="section upload-section">', unsafe_allow_html=True)
        st.subheader("📷 Upload a Plant Image")
        uploaded_image = st.file_uploader("Choose an image of a diseased plant leaf...", type=["jpg", "jpeg", "png"])
        st.markdown('</div>', unsafe_allow_html=True)

    # Result Section
    if uploaded_image is not None:
        image = Image.open(uploaded_image)

        with st.container():
            st.markdown('<div class="section">', unsafe_allow_html=True)
            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                st.image(image, caption="Uploaded Plant Image", use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                if st.button('Analyze Disease', key='analyze_button'):
                    with st.spinner('Analyzing your plant image...'):
                        # Get model prediction with confidence
                        prediction, confidence = predict_image_class(model_plant, image, class_indices)
                        confidence_percentage = confidence * 100

                        # Success message with confidence level and visual indicator
                        confidence_color = "#00897B" if confidence_percentage > 70 else "#FF8F00" if confidence_percentage > 40 else "#D32F2F"
                        st.markdown(f"""
                        <div style="padding: 1rem; background: linear-gradient(to right, {confidence_color}20, {confidence_color}10); 
                                    border-left: 5px solid {confidence_color}; border-radius: 8px; margin-bottom: 1rem;">
                            <h3 style="color: {confidence_color}; margin: 0;">Detected: {prediction}</h3>
                            <p style="margin: 0.5rem 0 0;">Confidence: {confidence_percentage:.1f}%</p>
                            <div class="progress-container">
                                <div class="progress" style="width: {confidence_percentage}%; background-color: {confidence_color};"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Create tabs for different information types
                        tabs = st.tabs(["Disease Details", "API Analysis", "Treatment Methods"])

                        # Get disease details only once to avoid multiple API calls
                        disease_details = get_disease_details(prediction)
                        treatment_details = get_treatment_details(prediction)

                        with tabs[0]:
                            st.markdown(f"""
                            <div class="disease-details">
                                {disease_details}
                            </div>
                            """, unsafe_allow_html=True)

                        with tabs[1]:
                            # Get API results
                            with st.spinner('Fetching analysis from Plant.id API...'):
                                disease_info = fetch_disease_info_from_api(image)
                                if disease_info:
                                    st.subheader("Alternative Analysis Results")
                                    for suggestion in disease_info['suggestions']:
                                        # Determine confidence class
                                        confidence_class = "confidence-high" if suggestion[
                                                                                    'probability'] > 70 else "confidence-medium" if \
                                        suggestion['probability'] > 40 else "confidence-low"
                                        confidence_color = "#00897B" if suggestion['probability'] > 70 else "#FF8F00" if \
                                        suggestion['probability'] > 40 else "#D32F2F"

                                        with st.expander(
                                                f"{suggestion['name']} ({suggestion['probability']:.1f}% confidence)"):
                                            st.markdown(f"""
                                                <div class="disease-card {confidence_class}">
                                                    <h4 style="color: {confidence_color};">{suggestion['name']}</h4>
                                                    <p><strong>Confidence:</strong> {suggestion['probability']:.1f}%</p>
                                                    <div class="progress-container">
                                                        <div class="progress" style="width: {min(100, suggestion['probability'])}%; background-color: {confidence_color};"></div>
                                                    </div>
                                                </div>
                                            """, unsafe_allow_html=True)
                                else:
                                    st.warning("Couldn't fetch API analysis. Please try again.")

                        with tabs[2]:
                            st.markdown(f"""
                            <div class="treatment-details">
                                <h3>💊 Treatment Methods for {prediction}</h3>
                                {treatment_details}
                            </div>
                            """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

elif app_mode == "Plant Care Chatbot":
    st.markdown('<div class="section chat-tab">', unsafe_allow_html=True)

    st.markdown("""
    <div class="header-section">
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="#2E7D32" viewBox="0 0 16 16" style="vertical-align: middle; margin-right: 10px;">
            <path d="M8 16A8 8 0 1 0 8 0a8 8 0 0 0 0 16zm.93-9.412-1 4.705c-.07.34.029.533.304.533.194 0 .487-.07.686-.246l-.088.416c-.287.346-.92.598-1.465.598-.703 0-1.002-.422-.808-1.319l.738-3.468c.064-.293.006-.399-.287-.47l-.451-.081.082-.381 2.29-.287zM8 5.5a1 1 0 1 1 0-2 1 1 0 0 1 0 2z"/>
        </svg>
       
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <p>Ask PlantMedic about plant diseases, gardening tips, or specific plant care questions. 
    The chatbot is powered by Google's Gemini AI and provides expert advice on plant health.</p>
    """, unsafe_allow_html=True)

    # Initialize session state for chat history if it doesn't exist
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = [
            {"role": "bot",
             "content": "👋 Hello! I'm PlantMedic, your plant care assistant. How can I help with your plants today?"}
        ]

    # Display chat messages
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f'<div class="chat-message user-message">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message bot-message">{message["content"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Chat input
    user_input = st.text_input("Type your plant-related question here:", key="user_query")

   if st.button("Send", key="send_button"):
        if user_input:
        # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Get bot response
            with st.spinner('PlantMedic is thinking...'):
                bot_response = get_chatbot_response(user_input)

        # Add bot response to chat history
            st.session_state.chat_history.append({"role": "bot", "content": bot_response})

        # Clear the input box
            st.rerun()  # Changed from st.experimental_rerun()

