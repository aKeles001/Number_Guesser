import pandas as pd
from PIL import Image
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model

drawing_mode = "freedraw"
stroke_width = 25 #st.sidebar.slider("Stroke width: ", 1, 25, 3)
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
realtime_update = st.sidebar.checkbox("Update in realtime", True)

@st.cache_resource
def get_model():
    model = load_model('model/model2.keras')
    return model

model = get_model()

canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color="#eee",
    update_streamlit=realtime_update,
    height=280,
    width=280,
    drawing_mode=drawing_mode,
    key="canvas",
)


def prepare_canvas_image(image_data):
    # Convert to grayscale
    img = Image.fromarray((image_data[:, :, :3]).astype(np.uint8)).convert("L")
    img = Image.fromarray(255 - np.array(img))
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    img = img.resize((20, 20), Image.LANCZOS)
    new_img = Image.new("L", (28, 28), 0)
    new_img.paste(img, ((28 - 20) // 2, (28 - 20) // 2))
    new_img = new_img.convert("RGB")
    arr = np.array(new_img).astype("float32") / 255.0
    arr = arr.reshape(1, 28, 28, 3)
    return arr

if canvas_result.image_data is not None:
    st.image(canvas_result.image_data)
    if st.button("Predict"):
        input_data = prepare_canvas_image(canvas_result.image_data)
        st.write(input_data.shape, input_data.dtype)
        prediction = model.predict(input_data)
        predicted_class = np.argmax(prediction)
        st.write(f"Predicted class: {predicted_class}")
        #st.write(f"Prediction probabilities: {prediction[0]}")