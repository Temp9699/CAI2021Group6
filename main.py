#import
import streamlit as st
from PIL import Image
from predict import predict, similar_image

# uploading image
st.header('Equipment Identification')
st.title("Predict New Images")
uploadedfile = st.file_uploader("Choose Image", type = ['jpg','png','jpeg'])

if uploadedfile is not None:
    # Make prediction
    image = Image.open(uploadedfile)
    image.save("images/tmp_image.jpg")
    prediction = predict("images/tmp_image.jpg")

    # Display uploaded image
    st.title("Uploaded image")
    st.image(uploadedfile, caption = 'uploaded image',width = 500)

    # Display similar images
    # st.title("Similar images")
    # similar1 = Image.open("CAI2\EDC\similar.jpg")
    # st.image(similar1 ,width = 500)
    # similar2 = Image.open("CAI2\EDC\similar2.jpg")
    # st.image(similar2 ,width = 500)

    # Display description
    st.title("Description")
    st.text_input("Enter some description")

    # Display predictions
    st.title("Predicton")
    for i in range(len(prediction)):
        st.checkbox(prediction[i])

    # Submit button
    st.button("Submit")
