import streamlit as st
import requests
import base64
import json

st.set_page_config(page_title="Predikcija žanra filma")

st.title("Predikcija žanra filma na temelju postera")
st.write("Uploadaj sliku filmskog postera i model će predvidjeti vjerojatne žanrove.")

ENDPOINT_URL = "https://genre-predictor-9655.italynorth.inference.ml.azure.com/score"

with st.sidebar:
    st.header("Postavke")
    api_key = st.text_input("Azure ML API kljuc", type="password")
    threshold = st.slider("Prag za predikciju zanra", 0.0, 1.0, 0.5, 0.05)

uploaded_file = st.file_uploader("Odaberi poster filma", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Ucitani poster", width=300)

    if st.button("Predvidi zanr"):
        if not api_key:
            st.error("Unesi API kljuc u lijevom izborniku.")
        else:
            with st.spinner("Saljem zahtjev modelu..."):
                image_bytes = uploaded_file.read()
                image_b64 = base64.b64encode(image_bytes).decode("utf-8")

                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                }
                payload = {"image": image_b64}

                try:
                    response = requests.post(ENDPOINT_URL, headers=headers, data=json.dumps(payload), timeout=30)

                    if response.status_code == 200:
                        raw = response.json()
                        result = json.loads(raw) if isinstance(raw, str) else raw

                        st.success("Predikcija gotova.")
                        st.subheader("Predvideni zanrovi:")
                        if result.get("predicted_genres"):
                            st.write(", ".join(result["predicted_genres"]))
                        else:
                            st.write("Model nije predvidio nijedan zanr iznad praga.")

                        st.subheader("Vjerojatnosti po zanru:")
                        probs = result.get("probabilities", {})
                        sorted_probs = dict(sorted(probs.items(), key=lambda x: x[1], reverse=True))
                        st.bar_chart(sorted_probs)
                    else:
                        st.error(f"Greska: {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"Greska u komunikaciji s endpointom: {e}")
