import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load YOLO model
model = YOLO("model/best.pt")

st.set_page_config(page_title="Cashew Nut Grading System", layout="centered")
st.title("Cashew Nut Grading – AI + Weight Based System")

# --- Inputs ---
uploaded_file = st.file_uploader("Upload cashew nut image", type=["jpg", "png"])
weight = st.number_input("Enter nut weight (grams)", min_value=0.0, step=0.1)

# --- Weight based grading ---
def grade_from_weight(weight):
    if weight >= 8.0:
        return "Grade_A"
    elif weight >= 5.0:
        return "Grade_B"
    else:
        return "Grade_C"

if uploaded_file and weight > 0:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict Grade"):
        img_array = np.array(image)

        # Run YOLO
        results = model(img_array, conf=0.25)

        # Get model prediction
        if len(results[0].boxes) > 0:
            cls_id = int(results[0].boxes[0].cls[0])
            model_grade = model.names[cls_id]
        else:
            model_grade = "No Detection"

        # Weight-based grade
        weight_grade = grade_from_weight(weight)

        # --- Final decision logic ---
        if model_grade == weight_grade:
            final_grade = model_grade
            decision = "Model and weight agree"
        else:
            final_grade = weight_grade
            decision = "Weight-based correction applied"

        # --- Display ---
        st.subheader("Results")
        st.write(f"🧠 Model Prediction: **{model_grade}**")
        st.write(f"⚖ Weight-Based Grade: **{weight_grade}**")
        st.success(f"✅ Final Grade: **{final_grade}**")
        st.info(decision)

        # Show annotated image
        annotated_img = results[0].plot()
        st.image(annotated_img, caption="Detection Output", use_container_width=True)
