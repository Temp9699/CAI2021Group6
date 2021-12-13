#import more here
import streamlit as st
from PIL import Image
from predict import predict

showresult = 0

st.header('Equipment Identification')

st.title("Predict New Images")
uploadedfile = st.file_uploader("Choose Image", type = ['jpg','png','jpeg'])

if st.button('Predict') and uploadedfile is not None:
    image = Image.open(uploadedfile)
    image.save("images/tmp_image.jpg")
    prediction = predict("images/tmp_image.jpg")
    showresult = 1

if showresult == 1:

    st.title("Result")
    st.image(uploadedfile, caption = 'uploaded image',width = 200, use_column_width=True)
    st.title("Prediction:")
    st.write(prediction)
    st.title("Description")
    st.text_input("Tell us about the problem")
    if st.button('submit'):
        showresult = 0
