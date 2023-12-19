from rembg import remove
from PIL import Image
from rembg import remove
from rembg.session_factory import new_session
import streamlit as st

def choose_model(model_choose):
    if model_choose == 'Human':
        model = 'u2net_human_seg'
    else:
        model = 'u2net'
    return model

def remove_background(input, model_choose='Human'):
    model = choose_model(model_choose)
    output = remove(input, alpha_matting_erode_size=15, session=new_session(model))
    return output

