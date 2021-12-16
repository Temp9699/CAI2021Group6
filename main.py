#import
import streamlit as st
from PIL import Image
from predict import predict, filenames

# uploading image
st.header('Equipment Identification')
st.title("Predict New Images")
uploadedfile = st.file_uploader("Choose Image", type = ['jpg','png','jpeg'])

if uploadedfile is not None:
    # Make prediction
    image = Image.open(uploadedfile)
    image = image.convert('RGB')
    image.save("images/tmp_image.jpg")
    prediction, indices = predict("images/tmp_image.jpg")

    # Display uploaded image
    st.title("Uploaded image")
    st.image(uploadedfile, caption = 'uploaded image',width = 500)

    # Display similar images
    st.title("Similar images")
    # similar1 = Image.open("CAI2\EDC\similar.jpg")
    # st.image(similar1 ,width = 500)
    # similar2 = Image.open("CAI2\EDC\similar2.jpg")
    # st.image(similar2 ,width = 500)
    for rank, idx in enumerate(indices):
        st.image(filenames[idx], caption = f"similar image: {rank}", width = 500)


    # Display description
    st.title("Description")
    st.text_input("Enter some description")

    # Display predictions
    st.title("Prediction")
    for i in range(len(prediction)):
        st.checkbox(prediction[i])
    # Submit button
    st.button("Submit")
