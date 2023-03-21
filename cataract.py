import streamlit as st
import tensorflow as tf
import numpy as np
from keras.applications.resnet import preprocess_input
from PIL import Image


#####functions#########
def prediction(modelname, sample_image, IMG_SIZE = (224,224)):

    #labels
    labels = ["Mild","Normal"]

    try:
        #loading the .h5 model
        load_model = tf.keras.models.load_model(modelname)

        sample_image = Image.open(sample_image)
        img_array = sample_image.resize(IMG_SIZE)
        img_batch = np.expand_dims(img_array, axis = 0)
        image_batch = img_batch.astype(np.float32)
        image_batch = preprocess_input(image_batch)
        prediction = load_model.predict(img_batch)
        return labels[int(np.argmax(prediction, axis = 1))]


    except Exception as e:
        st.write("ERROR: {}".format(str(e)))


#Building the website

#title of the web page
st.title("Cataract Image Predictior")

#setting the main picture
st.image(
    "https://columbiaeyeclinic.com/wp-content/uploads/2019/11/tech-eye-4-scaled.jpg", 
    caption = "Cataract Eyes")

#about the web app
st.header("About the Web App")

#details about the project
with st.expander("Web App 🌐"):
    st.subheader("Cataract Image Predictions")
    st.write("This web app is about.....................")

#setting the tabs
tab1, tab2 = st.tabs(['Image Upload 👁️', 'Camera Upload 📷'])

#tab1
with tab1:
    #setting file uploader
    #you can change the label name as your preference
    image = st.file_uploader(label="Upload an image",accept_multiple_files=False, help="Upload an image to classify them")

    if image:
        #validating the image type
        image_type = image.type.split("/")[-1]
        if image_type not in ['jpg','jpeg','png','jfif']:
            st.error("Invalid file type : {}".format(image.type), icon="🚨")
        else:
            #displaying the image
            st.image(image, caption = "Uploaded Image")

            #getting the predictions
            label = prediction("final_cataract.h5", image)

            #displaying the predicted label
            st.subheader("Your Condition is **{}**".format(label))

with tab2:
    #camera input
    cam_image = st.camera_input("Please take a photo")

    if cam_image:
        #displaying the image
        st.image(cam_image)

        #getting the predictions
        label = prediction("final_cataract.h5", cam_image)

        #displaying the predicted label
        st.subheader("Your Condition is **{}**".format(label))
            




