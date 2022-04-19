import os
import streamlit as st
from InferenceCode import Inference
from keras import backend as K



def main():
    st.markdown('<p style="font-size: 30px;"><strong>Liquid Volume Detection From Drain Images</strong></p>',
                    unsafe_allow_html=True)


    image_file = st.file_uploader("Upload An Image",type=['png','jpeg','jpg'])
    total_volume = st.number_input("Enter the total volume of drain",min_value=50,max_value=1000)
# DEMO_IMAGE = "C:\\Users\\slmcy\\Desktop\\Graduation_Project\\First_Train_Results\\Test_100ml\\50ml2.jpg"

    if image_file is not None:
        with open(image_file.name, mode="wb") as f:
            f.write(image_file.getbuffer())
        st.success("Volume Detection Started.")
        K.clear_session()
        CUSTOM_MODEL_PATH = "models\\drain20211219T1936\\mask_rcnn_drain_0012.h5"
        MODEL_DIR = "models\\drain20211219T1936"
        img = Inference(CUSTOM_MODEL_PATH, MODEL_DIR, image_file.name,total_volume)
        K.clear_session()
        os.remove(image_file.name)
        st.image(
            img, caption=f"Processed image", use_column_width=True,
        )





if __name__ == '__main__':
    main()