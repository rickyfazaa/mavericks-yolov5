import streamlit as st
import torch
from PIL import Image
from io import *
import glob
from datetime import datetime
import os
import wget
import cv2 # Lupa di-import
from streamlit import session_state
from video_predict import runVideo
# from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode
# st.set_page_config(layout="wide")
import subprocess


import numpy as np
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

latar = """
<style>
[class="main css-uf99v8 egzxvld5"]{
  background-image: url("https://telegra.ph/file/b99e84fd8c75391a1ebc9.png");
  background-repeat: no-repeat;
  background-position: center;
  background-size: cover;
}


[class="css-vk3wp9 e1fqkh3o11"]{
  background-image: url("https://telegra.ph/file/1aa88f365322b1972bf3f.png");
  background-repeat: no-repeat;
  background-position: center;
  background-size: cover;
}



[data-testid="stHeader"]{
  background-image: url("https://telegra.ph/file/9d4c3a7cbd1b60350d814.png");
  background-repeat: no-repeat;
  background-position: center;
  background-size: cover;
}
</style>
"""

st.markdown(latar, unsafe_allow_html=True)


# Configurations
CFG_MODEL_PATH = "best.pt"
CFG_ENABLE_URL_DOWNLOAD = False
CFG_ENABLE_VIDEO_PREDICTION = True
if CFG_ENABLE_URL_DOWNLOAD:
    # Configure this if you set cfg_enable_url_download to True
    url = "https://drive.google.com/uc?id=1OAU7ruLtsiMQE1krPRIRhYwcEkQdMLfD"
# End of Configurations


def imageInput(model, src):

    if src == 'Upload your own data.':
        image_file = st.file_uploader(
            "Upload An Image", type=['png', 'jpeg', 'jpg'])
        col1, col2 = st.columns(2)
        if image_file is not None:
            img = Image.open(image_file)
            with col1:
                st.image(img, caption='Uploaded Image',
                         use_column_width='always') # use_column_width='always' width=256
            ts = datetime.timestamp(datetime.now())
            imgpath = os.path.join('input', str(ts)+image_file.name)
            outputpath = os.path.join("output")
            with open(imgpath, mode="wb") as f:
                f.write(image_file.getbuffer())

            with st.spinner(text="Predicting..."):
                # Load model
                # cap = cv2.VideoCapture(imgpath)
                # ret, frame = cap.read()
                #frame = cv2.imread(imgpath)
                #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                #pred = model(frame)
                #pred.render()
                # save output to file
                #for im in pred.imgs:
                    #im_base64 = Image.fromarray(im)
                    #im_base64.save(outputpath)
                # subprocess.run("ls")
                subprocess.run(['python3', 'detect.py', '--weights', CFG_MODEL_PATH, '--img', '256', '--conf', '0.4', '--source', imgpath])

            # Predictions
            output_imgpath = os.path.join("output")
            img_ = Image.open(output_imgpath)
            with col2:
                st.image(img_, caption='Model Prediction(s)',
                         use_column_width='always') # use_column_width='always'


def videoInput(model, src):
    if src == 'Upload your own data.':
        uploaded_video = st.file_uploader(
            "Upload A Video", type=['mp4', 'mpeg', 'mov'])
        pred_view = st.empty()
        warning = st.empty()
        if uploaded_video != None:

            # Save video to disk
            ts = datetime.timestamp(datetime.now())  # timestamp a upload
            uploaded_video_path = os.path.join(
                'input', str(ts)+uploaded_video.name)
            with open(uploaded_video_path, mode='wb') as f:
                f.write(uploaded_video.read())

            # Display uploaded video
            with open(uploaded_video_path, 'rb') as f:
                video_bytes = f.read()
            st.video(video_bytes)
            st.write("Uploaded Video")
            submit = st.button("Run Prediction")
            if submit:
                runVideo(model, uploaded_video_path, pred_view, warning)

    elif src == 'From example data.':
        # Image selector slider
        videopaths = glob.glob('data/example_videos/*')
        if len(videopaths) == 0:
            st.error(
                'No videos found, Please upload example videos in data/example_videos', icon="⚠️")
            return
        imgsel = st.slider('Select random video from example data.',
                           min_value=1, max_value=len(videopaths), step=1)
        pred_view = st.empty()
        video = videopaths[imgsel-1]
        submit = st.button("Predict!")
        if submit:
            runVideo(model, video, pred_view, warning)

def mavericks():
        # RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
        if "myImage" not in session_state.keys():
            session_state['myImage'] = None
        # img_capture = webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION)
        #subprocess.run(['python3', 'detect.py', '--weights', CFG_MODEL_PATH, '--source', "0"])
        if img_capture:
            session_state['myImage'] = img_capture

        if session_state['myImage']:
            st.image(session_state['myImage'])
            #col1 = st.columns(1)
            #col1, col2, col3 = st.columns([1,1,1])
            #with col1:
            #    st.image(session_state['myImage'])
            #with col2:
            #    st.image(session_state['myImage'])
            #with col3:
            #    st.image(session_state['myImage'])

            colA, colB = st.columns(2)
            fileName = session_state['myImage'].name
            with colA:
                saveButton = st.button("Save Image")
                if saveButton:
                    with open(fileName, "wb") as imageFile:
                        imageFile.write(session_state['myImage'].getbuffer())
                        st.success("The file saved with name", fileName)

            with colB:
                st.download_button("Download Image", data=session_state['myImage'],file_name=fileName)

def webcam_streamlit():
        device = "cpu"
        st.model = torch.hub.load('rickyfazaa/yolov5', 'custom',
                           path=CFG_MODEL_PATH, force_reload=True, device=device, skip_validation=True)
        RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})


        class VideoProcessor:
            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                
                # vision processing
                flipped = img[:, ::-1, :]

                # model processing
                im_pil = Image.fromarray(flipped)
                results = st.model(im_pil, size=112)
                bbox_img = np.array(results.render()[0])

                return av.VideoFrame.from_ndarray(bbox_img, format="bgr24")


        webrtc_ctx = webrtc_streamer(
            key="WYH",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=False,
        )


def main():

    
    # -- Sidebar
    st.sidebar.title('⚙️ Options')

    if CFG_ENABLE_VIDEO_PREDICTION:
        option = st.sidebar.selectbox("Select Activity.", ['Home', 'Deteksi Melalui Image', 'Deteksi Melalui Video', 'Real-time'])
    else:
        option = st.sidebar.radio("Select Activity.", ['Home','Image'])

    datasrc = st.sidebar.radio("Select input source.", [
                               'Upload your own data.'],disabled=False)

    if torch.cuda.is_available():
        deviceoption = st.sidebar.radio("Select compute Device.", [
                                        'cpu', 'cuda'], disabled=False, index=0)
        
    else:
        deviceoption = st.sidebar.radio("Select compute Device.", [
                                        'cpu', 'cuda'], disabled=True, index=0)
    # -- End of Sidebar

    st.header('📦 Penerapan Teknologi Object Detection dalam Identifikasi Pemakaian APD pada Pekerja Konstruksi')
    st.sidebar.markdown(
        """![image](https://telegra.ph/file/445ce9e4b20d64ed84f02.png)

Developed by Mavericks    
Email : mvrteam11@gmail.com  
[Leader LinkedIn](https://id.linkedin.com/in/rickyfazaa)
[Leader GitHub](https://github.com/rickyfazaa)""")

    if option == "Deteksi Melalui Image":
        imageInput(loadmodel(deviceoption), datasrc)
    elif option == "Deteksi Melalui Video":
        videoInput(loadmodel(deviceoption), datasrc)
    elif option == "Real-time":
        webcam_streamlit()
    elif option == 'Home':
        html_temp_home1 = """<div style="background-color:#6D7B8D;padding:10px">
                                            <h4 style="color:white;text-align:center;">
                                            Helm, Vest, dan Person detection menggunakan YOLOv5 Custom Model dan Streamlit</h4>
                                            </div>
                                            </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)
        st.write("""
                 Aplikasi ini memiliki dua fungsi, yaitu

                 1. Fungsi untuk mendeteksi APD (Alat Pelindung Diri) pekerja konstruksi pada Gambar.

                 2. Fungsi untuk mendeteksi APD (Alat Pelindung Diri) pekerja konstruksi pada Video. 
                 
                 3. Fungsi untuk mendeteksi APD (Alat Pelindung Diri) pekerja konstruksi secara REAL-TIME Webcam. """)


# Downlaod Model from url.
@st.cache_resource
def downloadModel():
    if not os.path.exists(CFG_MODEL_PATH):
        wget.download(url, out="models/")
        

@st.cache_resource
def loadmodel(device):
    model = torch.hub.load('rickyfazaa/yolov5', 'custom',
                           path=CFG_MODEL_PATH, force_reload=True, device=device, skip_validation=True)
    return model


if __name__ == '__main__':
    main()
