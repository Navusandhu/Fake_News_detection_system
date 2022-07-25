import os
import pandas as pd
from PIL import Image
from requests import head
import streamlit as st
from model import FakeNewsClassifier
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="Fake-News-Classification",
    page_icon="👋",
)


def header(text, col="#ff6622", new_col=None):
    if new_col is None:
        st.markdown(
            f'<p style="background-color:{col};color:#ffffff;font-size:30px;border-radius:2%;">{text}</p>', unsafe_allow_html=True)
    else:
        new_col.markdown(
            f'<p style="background-color:{col};color:#ffffff;font-size:30px;border-radius:2%;">{text}</p>', unsafe_allow_html=True)


@st.cache(allow_output_mutation=True)
def load_fake_news_classifier_model(path="./utils"):
    model_obj = FakeNewsClassifier(path)
    return model_obj


model_obj = load_fake_news_classifier_model()

selected = option_menu(
    menu_title="Menu",
    options=['Reliability check', 'Plots & graphs'],
    icons=['home', 'book'],
    orientation="horizontal"
)


def reliability_checker():
    col1, col2 = st.columns((30, 20))
    col1.title("Check reliability of your Text")
    sentence = col1.text_area(
        'Please paste your Text :', height=360)
    button = col1.button("Check Reliability")
    with st.spinner("Discovering Answers.."):
        if button and sentence:
            prediction, probab = model_obj.get_prediction(sentence)
            for _ in range(4):
                col2.header('')
            col = "#ff6622"
            if prediction == "Reliable":
                col = "#44dd22"
            header(f"The text is: {prediction}", col)
            st.subheader(
                f"Reliable: {probab[0][0]*100:.2f}% | Unreliable: {probab[0][1]*100:.2f}%")

            chart_data = pd.DataFrame(
                [[float(f"{probab[0][0]*100:.2f}"), None],
                 [None, float(f"{probab[0][1]*100:.2f}")]],
                columns=["Reliable", "Unreliable"])
            col2.bar_chart(chart_data, width=400, use_container_width=False)

            labels = ["Reliable", "Unreliable"]
            sizes = [probab[0][0], probab[0][1]]
            explode = [0.1, 0]
            fig, ax = plt.subplots()
            ax.pie(sizes, explode=explode, labels=labels,
                   shadow=True, startangle=90, autopct="%0.0001f%%")
            ax.axis("equal")
            col2.pyplot(fig)


def show_plots_graphs():
    status = 0
    root_path = "./images/insights_and_outputs/"
    st.header(f"plots and graphs")
    col1, col2 = st.columns(2)
    columns = [col1, col2]
    col_val = 0

    for folder in os.listdir(root_path):
        if os.path.isdir(os.path.join(root_path, folder)):
            if 'cloud' in folder:
                status = 1
            if status:
                st.subheader(folder.replace('_', ' '))
            plots_path = os.listdir(os.path.join(root_path, folder))
            for each_image in plots_path:
                image_path = os.path.join(root_path, folder, each_image)
                image = Image.open(image_path)
                width, height = image.size
                if status:
                    # Setting the points for cropped image
                    left = 20
                    top = 600
                    right = 1500
                    bottom = 2.75 * height / 4
                    bbox = (left, top, right, bottom)
                    image = image.crop(bbox)
                col = columns[col_val]
                if status:
                    st.image(image, caption=image_path.split(
                        '/')[-1].replace('.png', '') + f"col: {col_val}")
                else:
                    col.image(image, caption=image_path.split(
                        '/')[-1].replace('.png', '') + f"col: {col_val}")
                    col_val = (col_val + 1) % 2
            if not status:
                col1.markdown(
                    """---""")


if selected == "Reliability check":
    reliability_checker()
if selected == "Plots & graphs":
    show_plots_graphs()
