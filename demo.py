import streamlit as st
from model import remove_background
from PIL import Image
import requests
from io import BytesIO
import imageio.v2 as imageio
from super_image import EdsrModel, ImageLoader

st.title("Remove background demo")

def read_img(input_image):
    return Image.open(input_image)

def display_images(input_image, source='upload', is_read=True, model_choose='Human'):
    col1, col2 = st.columns(2)
    with col1:
        caption = "Image from {}".format(source)
        st.image(input_image, caption=caption, use_column_width=True)
    with col2:
        if is_read:
            input_image = read_img(input_image)
        output = remove_background(input_image, model_choose)
        st.image(output, caption="Remove background image", use_column_width=True)

def upscale_image(input_image):
    model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=2)
    inputs = ImageLoader.load_image(input_image)
    preds = model(inputs)
    ImageLoader.save_image(preds, './scaled_2x.png')

with st.sidebar:
    uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])

    picture = st.camera_input("Take a picture")

    url = st.text_input('Enter image url')
    model_choose = st.radio(
    "Choose a model",
    ["Human", "Object"])
    

if uploaded_file is not None:
    display_images(uploaded_file, source='upload', is_read=True, model_choose=model_choose)

if picture is not None:
    display_images(picture, source='camera', is_read=True, model_choose=model_choose)

if url is not None:
    try:
        response = requests.get(url)
        response.raise_for_status()  

        image = imageio.imread(BytesIO(response.content))
        url_image = Image.fromarray(image)

        display_images(url_image, source='url', is_read=False, model_choose=model_choose)
    except Exception as e:
        print(e)
            
