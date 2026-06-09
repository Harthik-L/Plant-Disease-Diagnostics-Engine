import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="AI Leaf Doctor", layout="centered", page_icon="🌿")
st.title("🌿 AI Plant Disease Diagnostics Engine")
st.write("Upload a high-resolution leaf snapshot to evaluate plant health conditions over backend APIs.")

# Targeting local FastAPI deployment instance
API_ENDPOINT = "http://127.0.0.1:8000/predict"

uploaded_file = st.file_uploader("Choose a leaf image file...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Target Sample', use_column_width=True)
    
    if st.button("Analyze Leaf Health", type="primary"):
        with st.spinner('Forwarding metrics to FastAPI Inference Engine...'):
            try:
                # Convert PIL image file target into byte arrays for request transmission
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='JPEG')
                img_byte_arr = img_byte_arr.getvalue()
                
                # Format request payload structure
                files = {'file': ('image.jpg', img_byte_arr, 'image/jpeg')}
                
                # Dispatch network execution call to backend api microservice
                response = requests.post(API_ENDPOINT, files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    prediction = result["prediction"]
                    confidence = result["confidence"]
                    
                    if "HEALTHY" in prediction:
                        st.success(f"### Diagnosis: **{prediction}** ({confidence}% Confidence)")
                    else:
                        st.error(f"### Diagnosis Detected: **{prediction}** ({confidence}% Confidence)")
                        st.warning("📋 **Recommendation:** Isolate the plant variant and apply matching organic treatments or target root zones immediately.")
                else:
                    st.error(f"❌ API Server returned error: {response.status_code} - {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("❌ Connection Refused! Is your FastAPI server running? Run `python src/api.py` first.")