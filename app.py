import torch
import pickle
from model.model import TranslationModel, device
import streamlit as st
from decode import decode


def sidebar():
    tg_url = 'https://t.me/whatisslove7'
    gh_url = 'https://github.com/whatisslove11'
    repo_url = 'https://github.com/whatisslove11/translator-app'
    with st.sidebar:
        st.markdown(
            "## Руководство по использованию:\n"
            "1. Введите текст на немецком в первую графу\n"
            "2. После ввода нажмите enter\n"
            "3. Перевод вашего текста появится автоматически!\n"
        )

        st.markdown("---")
        st.markdown("# О проекте")
        st.markdown(
            "📖 Translator from scratch — проект, позволяющий"
            " понять, что мир сложных технологий не так сложен"
            " и далек. Благодаря этому open-source проекту вы можете"
            " как поэкспериментировать со встроенной моделью, так"
            " и изучить from zero устройство архитектуры Transformer\n\n"
            "**Note:** эта реализация не очень качественная в плане кода и перевода\n"
            "Качественную реализацию автор написал, но не успел обучить"
        )
        st.markdown(
            "Проект находится в стадии разработки. "
        )
        st.markdown("---")
        st.markdown("# Контакты:")
        st.markdown("Telegram: [click](%s)" % tg_url)
        st.markdown("Author's GitHub: [click](%s)" % gh_url)
        st.markdown("GitHub repo: [click](%s)" % repo_url)

def clear_submit():
    st.session_state["submit"] = False


st.set_page_config(page_title="Translator from scratch", page_icon="📖", layout="wide")
st.header("📖 Translator from scratch")

hide_default_format = """
       <style>
       div[role='textbox'] span{{white-space: pre-wrap}}
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       .stTextInput input {
        height: 200px;
        text-align: left;
        white-space: pre-wrap;
        word-break: break-word;
        padding-top: 0px;
        padding-bottom: 150px;
        }   
       </style>
       """

sidebar()

with open('de_dict.pkl', 'rb') as f:
    word2index = pickle.load(f)  # word2index

with open('en_dict.pkl', 'rb') as f:
    index2word = pickle.load(f)  # index2word

len_de = 32318
len_en = 21524

model_name = 'model.pt'

model = TranslationModel(len_de, len_en).to(device)
model.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))

# Store the initial value of widgets in session state
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

col2, col3 = st.columns(2)

with col2:
    text_input = st.text_input(
        "German",
        label_visibility=st.session_state.visibility,
        disabled=st.session_state.disabled,
        placeholder=None,
        max_chars=100,
    )

    with col3:
        # print(decode(text_input, model, word2index, index2word))
        txt = st.text_area(
            label='English',
            value=decode(text_input, model, word2index, index2word),
            height=200
        )
        # st.caption('Balloons. Hundreds of them...')

st.markdown(hide_default_format, unsafe_allow_html=True)