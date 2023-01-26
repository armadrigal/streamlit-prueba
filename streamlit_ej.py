import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import ml_model

st.title("PROYECTO GRUPAL DTS05")
st.markdown('### Analisis de sentimientos en reviews')
title = st.text_input('Escribe tu review en portugues', '')
image_positivo = Image.open('positivo.PNG')
image_negativo = Image.open('negativo.PNG')
if title != '':
    pred = ml_model.predictions([title])
    if pred[0,0] > 0.5:
        st.markdown('#### Sentimiento positivo')
        st.image(image_positivo, caption='')
    else:
        st.markdown('##### Sentimiento negativo')
        st.image(image_negativo, caption='')