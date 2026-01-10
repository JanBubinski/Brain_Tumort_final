import streamlit as st

import requests
from PIL import Image

My_URL="http://127.0.0.1:8000/classify"
st.title("My medical systems")
uploaded_file = st.file_uploader("Podaj plik",type=["jpg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image)
    with st.spinner("WGRANY PLIK"):
        files = {"file": uploaded_file}
        response = requests.post(My_URL, files)
        wynik = response.json()
        st.write(wynik)
        st.success(f"Klasa: {wynik.get('Klasa', 'Brak danych')}")
        st.info(f"Pewność: {wynik.get('Pewność', 'Brak danych')}")

