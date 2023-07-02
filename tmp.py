import streamlit as st


def sidebar():
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
            "📖 Translator from scratch - проект,  "
            "бла бла бла для идите нахуй адал кошу цкоаш ацуац "
            "тпа охуееннй лорем инспум, но вообще ни разу не лорем инспум. "
        )
        st.markdown(
            "Проект находится в стадии разработки. "
        )
        st.markdown("---")
        st.markdown("# Контакты:")


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
        txt = st.text_area(
            label='English',
            value=text_input,
            height=200
        )
        st.caption('Balloons. Hundreds of them...')

st.markdown(hide_default_format, unsafe_allow_html=True)