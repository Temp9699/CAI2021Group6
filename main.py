#import more here
import streamlit as st

showresult = 0

# load model

st.header('Equipment Identification')

st.title("Predict New Images")
uploadedfile = st.file_uploader("Choose Image", type = ['jpg','png','jpeg'])
if st.button('Predict') and uploadedfile is not None:
    showresult = 1
    # make prediction



if showresult == 1:
#     un-comment this after implementing AI
#     st.image(predictedimage, caption=equipmentname) 
#     st.title("Commonly seen damages")
#     check1 = st.checkbox("placeholder1")
#     check2 = st.checkbox("placeholder2")
#     check3 = st.checkbox("placeholder3")
#     check3 = st.checkbox("others")
    st.title("Result")
    st.write("Image placeholder")
    st.write("Predicted equipment name placeholder")
    st.title("Description")
    st.text_input("Enter some description")
    if st.button('submit'):
        showresult = 0