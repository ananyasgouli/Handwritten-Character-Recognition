import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# -------------------------------
# 1. Page Configuration
# -------------------------------

st.set_page_config(
    page_title="Handwritten Character Recognition",
    page_icon="✍️",
    layout="wide"
)

# -------------------------------
# 2. Custom Styling
# -------------------------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #100d24, #211442, #101b35);
    color: white;
}

h1, h2, h3, p, label {
    color: white !important;
}
[data-testid="stFileUploader"] {
    color: white !important;
}

[data-testid="stFileUploader"] section {
    background-color: #292044 !important;
    border: 1px dashed #b99aff !important;
}

[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span {
    color: white !important;
}

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #c4a7ff;
}

.subtitle {
    text-align: center;
    color: #d1c8e8;
    font-size: 17px;
}

.result-card {
    background: rgba(255,255,255,0.09);
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.2);
}

.result-character {
    font-size: 75px;
    font-weight: bold;
    color: #c4a7ff;
}

section[data-testid="stSidebar"] {
    background-color: #17112c;
}
.supported-characters,
.stMarkdown,
.stCaption {
    color: #ffffff !important;
}

[data-testid="stFileUploader"] small {
    color: #ffffff !important;
}
[data-testid="stFileUploader"] section {
    background-color: #292044 !important;
    border: 1px dashed #b99aff !important;
}

[data-testid="stFileUploader"] section:hover {
    background-color: #392b5c !important;
}

[data-testid="stFileUploader"] button {
    background-color: #7652a8 !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# 3. Load Character Model
# -------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "handwritten_character_model.keras"
    )

model = load_model()

# -------------------------------
# 4. Application Heading
# -------------------------------

st.markdown(
    '<div class="main-title">✍️ Handwritten Character Recognition</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">An AI-powered system to recognize handwritten digits and alphabets</div>',
    unsafe_allow_html=True
)

st.write("")
st.divider()

# -------------------------------
# 5. About Project
# -------------------------------

with st.expander("📌 About This Project"):
    st.write("""
    This project uses a Convolutional Neural Network (CNN)
    trained using the MNIST and EMNIST datasets.

    It recognizes:
    - Digits: 0–9
    - Alphabets: A–Z

    The model contains 36 output classes.
    """)

# -------------------------------
# 6. Input Section
# -------------------------------

st.subheader("🖊️ Try It Yourself")

left, right = st.columns(2)

with left:
    st.write("### Draw a Character")

    canvas_result = st_canvas(
        fill_color="rgba(255,255,255,1)",
        stroke_width=15,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="character_canvas",
        return_image_data=True

    )

with right:
    st.write("### Or Upload an Image")

    uploaded_file = st.file_uploader(
        "Upload a handwritten digit or letter",
        type=["png", "jpg", "jpeg"]
    )

    st.markdown(
    "<div style='background-color:#292044; color:white; padding:12px; "
    "border-radius:10px; font-size:16px; font-weight:bold;'>"
    "Supported Characters: 0–9 and A–Z</div>",
    unsafe_allow_html=True
)

st.divider()

# -------------------------------
# 7. Preprocessing Function
# -------------------------------

def preprocess_image(img):

    img = ImageOps.grayscale(img)

    # Convert black writing on white background
    # into white writing on black background
    img = ImageOps.invert(img)

    img_array = np.array(img)

    # Remove background noise
    img_array[img_array < 30] = 0

    coords = np.argwhere(img_array > 30)

    if len(coords) == 0:
        return None

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    cropped = img.crop((
        x_min, y_min,
        x_max + 1, y_max + 1
    ))

    cropped.thumbnail((20, 20))

    final_img = Image.new("L", (28, 28), 0)

    x = (28 - cropped.width) // 2
    y = (28 - cropped.height) // 2

    final_img.paste(cropped, (x, y))

    data = np.array(final_img).astype("float32") / 255.0
    data = data.reshape(1, 28, 28, 1)

    return final_img, data

# -------------------------------
# 8. Prediction
# -------------------------------

if st.button("🔍 Recognize Character", type="primary"):

    input_image = None

    if uploaded_file is not None:
        input_image = Image.open(uploaded_file)

    elif canvas_result.image_data is not None:
        input_image = Image.fromarray(
            canvas_result.image_data.astype("uint8")
        )

    if input_image is None:
        st.warning("Please draw or upload a character first.")

    else:
        processed = preprocess_image(input_image)

        if processed is None:
            st.warning("No handwriting detected. Please try again.")

        else:
            final_img, data = processed

            predictions = model.predict(data, verbose=0)[0]

            predicted_index = int(np.argmax(predictions))

            if predicted_index < 10:
                character = str(predicted_index)
            else:
                character = chr(
                    ord("A") + predicted_index - 10
                )

            confidence = float(
                predictions[predicted_index]
            ) * 100

            st.divider()

            result_col, image_col = st.columns(2)

            with result_col:
                st.markdown(
                    f"""
                    <div class="result-card">
                        <h3>Predicted Character</h3>
                        <div class="result-character">
                            {character}
                        </div>
                        <h4>Confidence: {confidence:.2f}%</h4>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with image_col:
                st.subheader("Processed Image")
                st.image(final_img, width=160)

            st.subheader("🏆 Top 3 Predictions")

            top_3 = np.argsort(predictions)[-3:][::-1]

            for index in top_3:

                if index < 10:
                    label = str(index)
                else:
                    label = chr(ord("A") + index - 10)

                score = float(predictions[index]) * 100

                st.write(f"**{label} — {score:.2f}%**")
                st.progress(float(predictions[index]))

# -------------------------------
# 9. Footer
# -------------------------------

st.divider()

st.markdown(
    "<p style='text-align:center;color:#c4a7ff;'>"
    "Developed using Python | TensorFlow | CNN | Streamlit"
    "</p>",
    unsafe_allow_html=True
)