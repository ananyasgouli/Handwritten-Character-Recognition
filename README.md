Handwritten Character Recognition

Project Overview

This project recognizes handwritten digits (0–9) and English alphabets (A–Z) using a Convolutional Neural Network (CNN) and provides an interactive interface built with Streamlit.

Users can draw a character on a digital canvas or upload an image to get a prediction.

Features

- Recognizes handwritten digits (0–9).
- Recognizes handwritten uppercase alphabets (A–Z).
- Interactive drawing canvas.
- Upload handwritten character images.
- Displays predicted characters and top predictions with confidence scores.
- User-friendly Streamlit interface.

Technologies Used

- Python
- TensorFlow / Keras
- CNN (Convolutional Neural Network)
- NumPy
- Pillow
- Streamlit
- Streamlit Drawable Canvas
- MNIST and EMNIST datasets

Model Performance

- 36 output classes (10 digits and 26 letters).
- Test accuracy: approximately 94.52%.

How to Run the Project

1. Clone this repository:
   
   git clone YOUR_REPOSITORY_URL

2. Open the project folder:
   
   cd Handwritten-Character-Recognition

3. Install dependencies:
   
   pip install -r requirements.txt

4. Run the application:
   
   python -m streamlit run app.py

Model

The trained CNN model is saved as "handwritten_character_model.keras".

Internship

Project developed as part of the CodeAlpha internship.

Author

Ananya Gouli

GitHub: https://github.com/ananyasgouli
