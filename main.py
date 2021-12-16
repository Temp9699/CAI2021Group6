#import
import streamlit as st
from PIL import Image
from predict import predict

# uploading image
st.header('Equipment Identification')
st.title("Predict New Images")
uploadedfile = st.file_uploader("Choose Image", type = ['jpg','png','jpeg'])
#similar images
filenames = ["datasets/cai2/cold drink freezer/20211124_114324(0).jpg",
            "datasets/cai2/cold drink freezer/20211124_114324(0).jpg",
             "datasets/cai2/cold drink freezer/20211124_114326(0).jpg",
             "datasets/cai2/cold drink freezer/20211124_114327(0).jpg",
             "datasets/cai2/cold drink freezer/20211124_114328.jpg",
            "datasets/cai2/cold drink freezer/20211124_114330(0).jpg",
            "datasets/cai2/cold drink freezer/20211124_114331.jpg",
            "datasets/cai2/cold drink freezer/20211124_114332.jpg",
            "datasets/cai2/cold drink freezer/20211124_114334.jpg",
            "datasets/cai2/cold drink freezer/20211124_114335.jpg",
            "datasets/cai2/cold drink freezer/20211124_114336.jpg",
            "datasets/cai2/cold drink freezer/20211124_114338.jpg",
            "datasets/cai2/cold drink freezer/20211124_114340(0).jpg",
            "datasets/cai2/cold drink freezer/20211124_114341(0).jpg",
            "datasets/cai2/cold drink freezer/20211124_114342.jpg",
            "datasets/cai2/cold drink freezer/20211124_114345(0).jpg",
            "datasets/cai2/cold drink freezer/20211124_114347(0).jpg",
            "datasets/cai2/EDC/20211210_123243_001 - Copy.jpg",
             "datasets/cai2/EDC/20211210_123243_004.jpg",
             "datasets/cai2/EDC/20211210_123243_007.jpg",
             "datasets/cai2/EDC/20211210_123243_011.jpg",
             "datasets/cai2/EDC/20211210_123243_012.jpg",
             "datasets/cai2/EDC/20211210_123243_013.jpg",
             "datasets/cai2/EDC/20211210_123243_014.jpg",
             "datasets/cai2/EDC/20211210_123243_015.jpg",
             "datasets/cai2/EDC/20211210_123243_016.jpg",
             "datasets/cai2/EDC/20211210_123243_017.jpg",
             "datasets/cai2/EDC/20211210_123243_018.jpg",
             "datasets/cai2/EDC/20211210_123243_019.jpg",
             "datasets/cai2/EDC/20211210_123243_020.jpg",
             "datasets/cai2/EDC/20211210_123243_030.jpg",
             "datasets/cai2/EDC/20211210_123243_024.jpg",
             "datasets/cai2/EDC/20211210_123243_025.jpg",
             "datasets/cai2/EDC/20211210_123243_023.jpg",
             "datasets/cai2/EDC/20211210_123243_022.jpg",
             "datasets/cai2/EDC/20211210_123243_027.jpg",
             "datasets/cai2/EDC/20211210_123243_021.jpg",
             "datasets/cai2/EDC/20211210_123243_003.jpg",
             "datasets/cai2/EDC/20211210_123243_002.jpg",
             "datasets/cai2/EDC/20211210_123243_009.jpg",
             "datasets/cai2/ice maker/IMG_25641210_122731_Burst01.jpg",
             "datasets/cai2/ice maker/IMG_25641210_122740_Burst01.jpg",
             "datasets/cai2/ice maker/IMG_25641210_122823_Burst07.jpg",
             "datasets/cai2/ice maker/IMG_25641210_122823_Burst12.jpg",
             "datasets/cai2/ice maker/IMG_25641210_122823_Burst15.jpg",
             "datasets/cai2/ice maker/IMG_25641210_122823_Burst17.jpg",
             "datasets/cai2/ice maker/IMG_25641210_122823_Burst19.jpg",
             "datasets/cai2/ice maker/IMG_25641210_122936_Burst02.jpg",
             "datasets/cai2/ice maker/IMG_25641210_122936_Burst05.jpg",
             "datasets/cai2/ice maker/IMG_25641210_122936_Burst08.jpg",
             "datasets/cai2/ice maker/IMG_25641210_122936_Burst08.jpg",
             "datasets/cai2/ice maker/IMG_25641210_123026_Burst02.jpg",
             "datasets/cai2/ice maker/IMG_25641210_123026_Burst03.jpg",
             "datasets/cai2/ice maker/IMG_25641210_123026_Burst06.jpg",
             "datasets/cai2/ice maker/IMG_25641210_123026_Burst09.jpg",
             "datasets/cai2/ice maker/IMG_25641210_123026_Burst10.jpg",
             "datasets/cai2/ice maker/IMG_25641210_123026_Burst13.jpg",
             "datasets/cai2/ice maker/IMG_25641210_123026_Burst15.jpg",
             "datasets/cai2/ice maker/IMG_25641210_123026_Burst17.jpg",
             "datasets/cai2/ice maker/IMG_25641210_123026_Burst18.jpg",
             "datasets/cai2/ice maker/IMG_25641210_123026_Burst19.jpg",
            ]

if uploadedfile is not None:
    # Make prediction
    image = Image.open(uploadedfile)
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
    st.title("Predicton")
    for i in range(len(prediction)):
        st.checkbox(prediction[i])

    # Submit button
    st.button("Submit")
