import html
import tkinter as tk
from io import StringIO
from pathlib import Path
from tkinter import filedialog
from typing import List

import streamlit as st
import torch
from htbuilder import H, styles
from htbuilder.units import unit

from methods.bert.generate import create_model, generate_question
from similarity import bert, tfidf

div = H.div
span = H.span
px = unit.px
rem = unit.rem
em = unit.em

root = tk.Tk()
root.withdraw()
root.wm_attributes('-topmost', 1)

# State data
if 'idfdir' not in st.session_state:
    st.session_state['idfdir'] = ''
if 'text_lines' not in st.session_state:
    st.session_state['text_lines'] = ''
if 'aqg_model' not in st.session_state:
    st.session_state['aqg_model'] = 'mrm8488/bert2bert_shared-portuguese-question-generation'
if 'sim_model' not in st.session_state:
    st.session_state['sim_model'] = 'ricardo-filho/bert-portuguese-cased-nli-assin-assin-2'
if 'text_highlight' not in st.session_state:
    st.session_state['text_highlight'] = list()
if 'auto_question' not in st.session_state:
    st.session_state['auto_question'] = ''


def highlight_text(text: List[str], hl_idx: List[int]):
    out = div()

    for i, sent in enumerate(text):
        bg = '#21c354' if i in hl_idx else None
        out((
            span(
                style=styles(
                    background=bg,
                    border_radius=rem(0.33),
                    padding=(rem(0.125), rem(0.5)),
                    overflow='hidden'
                )
            )(
                html.escape(sent + ' ')
            )
        ))
    return str(out)


def clear_state():
    st.session_state['idfdir'] = ''
    st.session_state['text_highlight'] = list()
    st.session_state['auto_question'] = ''


# Upload and write input file
uploaded_file = st.file_uploader('Carregue um ficheiro')
if uploaded_file is not None:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    text = stringio.read()
    st.session_state['text_lines'] = [s for s in text.splitlines(keepends=False)
                                      if s.rstrip()]
    st.markdown(highlight_text(st.session_state['text_lines'],
                               st.session_state['text_highlight']),
                unsafe_allow_html=True)

# Sidebar - Model parameters
with st.sidebar:
    answer = st.text_area('Respostas (uma por linha)')
    method = st.selectbox('Selecione um método', ['TF-IDF', 'BERT'],
                          on_change=clear_state)

    # Select corpus for TF-IDF
    if method == 'TF-IDF':
        st.write('Selecione um corpus para calcular o IDF')
        clicked = st.button('Selecionar pasta...')
        if clicked:
            st.session_state['idfdir'] = filedialog.askdirectory(master=root)

        if st.session_state['idfdir'] is not None:
            st.text_area('Pasta selecionada:',
                         st.session_state['idfdir'],
                         disabled=True)

    # Run model
    run = st.button('Executar')
    if run:
        tokenizer, aqg = create_model(st.session_state['aqg_model'])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if method == 'TF-IDF':
            context_idx = tfidf.create_context(Path(st.session_state['idfdir']),
                                               st.session_state['text_lines'],
                                               [answer])
        if method == 'BERT':
            context_idx = bert.create_context(st.session_state['text_lines'],
                                              [answer],
                                              st.session_state['sim_model'])
        st.session_state['text_highlight'] = context_idx.tolist()[0]
        context = [st.session_state['text_lines'][i]
                   for i in st.session_state['text_highlight']]
        ctext = ' '.join(context)
        auto_question, _ = generate_question(ctext, answer,
                                             tokenizer, aqg,
                                             device)
        st.session_state['auto_question'] = auto_question
        st.experimental_rerun()

    if st.session_state['auto_question']:
        st.success(st.session_state['auto_question'])
