# app.py - Streamlit demo (CPU, no training at startup)
import sys
from pathlib import Path

import streamlit as st

ARTIFACT_DIR = Path(__file__).resolve().parent / "normalizer_model"
REQUIRED_FILES = ["best_model.pt", "tokenizer.json", "config.json", "model_def.py", "inference.py"]

st.set_page_config(page_title="Informal Text Normalizer", page_icon="💬", layout="wide")

st.markdown("""
<style>
/* Input text */
.stTextArea textarea {
    color: white !important;
    -webkit-text-fill-color: white !important;
}

/* Text inside the Original and Normalized output boxes */
.stAlert p {
    color: white !important;
    -webkit-text-fill-color: white !important;
}

/* Placeholder text */
.stTextArea textarea::placeholder {
    color: #cccccc !important;
    -webkit-text-fill-color: #cccccc !important;
}
</style>
""", unsafe_allow_html=True)

st.title("💬 Informal Text Normalizer")
st.markdown(
    "Transforme des messages informels (Twitter / chat) en anglais normalisé grâce à un modèle "
    "**Seq2Seq GRU + attention** entraîné sur tokenisation **BPE**.  \n"
    "*Turns informal Twitter/chat English into normalized English (BPE + GRU Seq2Seq with attention).*"
)

missing = [name for name in REQUIRED_FILES if not (ARTIFACT_DIR / name).exists()]
if missing:
    st.error(
        f"Fichiers du modèle introuvables dans `{ARTIFACT_DIR}` : {', '.join(missing)}.\n\n"
        "Téléversez le dossier `normalizer_model/` (issu du notebook Kaggle) à côté de `app.py`."
    )
    st.stop()

@st.cache_resource(show_spinner="Chargement du modèle...")
def get_normalizer():
    sys.path.insert(0, str(ARTIFACT_DIR))
    from inference import load_normalizer
    return load_normalizer(ARTIFACT_DIR, device="cpu")

normalize_text = get_normalizer()

EXAMPLES = [
    "thaaaaaaaaank u for makin me happppy",
    "i need help cuz i am lost, so pls call me b4 u leave, thx",
    "ur a good man ,thx for your help",
    "omg i luv this song !",
    "send me the pic asap cuz idk if ppl like it, thx",
    "i want ma money naw"
]

if "user_text" not in st.session_state:
    st.session_state["user_text"] = EXAMPLES[0]

st.subheader("Exemples")
columns = st.columns(len(EXAMPLES))
for column, example in zip(columns, EXAMPLES):
    if column.button(example, use_container_width=True):
        st.session_state["user_text"] = example

user_text = st.text_area("Message informel", key="user_text", height=120)

if st.button("Normalize", type="primary"):
    if not user_text.strip():
        st.warning("Veuillez saisir un message.")
    else:
        result = normalize_text(user_text)
        left, right = st.columns(2)
        left.markdown("#### Original")
        left.info(user_text)
        right.markdown("#### Normalisé")
        right.success(result if result else "(sortie vide)")

st.caption("Projet universitaire - Youssef Nafae - Normalisation de Texte Informel")
