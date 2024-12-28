# image_question_answer_app.py
import streamlit as st
from PIL import Image
from image_question_answer_model import load_model, generate_response   # Importing the model logic from image_question_answer_model.py

# Initialize model and processor
model, processor = load_model()

# Streamlit app title
st.title("Visual Question Answering with Llava")

# Upload an image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Ask a question
    question = st.text_input("Enter your question about the image:")

    # Submit button
    if st.button("Submit"):
        if question:
            # Generate response using the model
            response = generate_response(model, processor, image, question)
            
            # To remove unwanted responses
            remove_text = response.find("ASSISTANT:")
            if remove_text != -1:
               answer = response[remove_text+11:]
            else:
               answer = "Sorry, not able to read the image"

            # Display the answer
            st.write(f"**Answer:** {answer}")
        else:
            st.write("Please enter a question.")