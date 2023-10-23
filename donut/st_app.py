import streamlit as st
from pathlib import Path

from donut import DonutModel
from predict import predict

root_path = Path("./result/train_cord")

model_name = st.selectbox(
    label="Choose a model", options=["donutocr_v1", "donutocr_background_removed_v1"]
)

model_path = root_path / model_name

model = DonutModel.from_pretrained(str(model_path))

remove_background = st.checkbox(label="remove background")

if model:
    img = st.file_uploader("Upload an image")

success = st.button(
    "Extract",
    on_click=predict,
    args=[model, img, remove_background],
)
