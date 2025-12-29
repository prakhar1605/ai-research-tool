import streamlit as st
import os
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --------------------
# Setup
# --------------------

st.set_page_config(
    page_title="AI Research Paper Explorer",
    layout="wide"
)

# --------------------
# CSS FIX (center + full width)
# --------------------
st.markdown("""
<style>
.block-container {
    padding-left: 5rem;
    padding-right: 5rem;
    max-width: 1400px;
    margin: auto;
}
.explain-box {
    padding: 2rem;
    border-radius: 14px;
    background-color: rgba(255,255,255,0.04);
    line-height: 1.75;
    font-size: 1.05rem;
}
</style>
""", unsafe_allow_html=True)

st.title("📚 AI Research Paper Explorer")
st.write("Learn famous AI research papers with **controlled depth and length**.")

st.markdown("---")

# --------------------
# LLM (OpenRouter)
# --------------------
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.3
)


parser = StrOutputParser()

# --------------------
# Controls
# --------------------
left, right = st.columns(2)

with left:
    word_limit = st.selectbox(
        "📏 Explanation length",
        ["500 words", "1000 words"]
    )

with right:
    depth = st.selectbox(
        "🧠 Explanation depth",
        ["Beginner", "Intermediate", "Deep"]
    )

words = "500" if "500" in word_limit else "1000"

# --------------------
# Prompt
# --------------------
prompt = PromptTemplate(
    template="""
You are a skilled AI tutor.

Explain the research paper below in approximately {words} words.

Depth level: {depth}

Guidelines:
- Beginner: intuition, motivation, simple examples
- Intermediate: architecture, working flow, pros/cons
- Deep: architecture details, training strategy, impact, limitations

Paper:
{paper}
""",
    input_variables=["paper", "words", "depth"]
)

chain = prompt | llm | parser

# --------------------
# Papers
# --------------------
PAPERS = {
    "Computer Vision": [
        "ImageNet Classification with Deep Convolutional Neural Networks (AlexNet, 2012)",
        "Deep Residual Learning for Image Recognition (ResNet, 2015)"
    ],
    "Sequence Models": [
        "Long Short-Term Memory (LSTM)",
        "Sequence to Sequence Learning with Neural Networks"
    ],
    "Generative Models": [
        "Generative Adversarial Networks (GANs)",
        "Variational Autoencoders (VAE)"
    ],
    "Transformers / NLP": [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers"
    ],
    "Others (Custom)": []
}

category = st.selectbox("📂 Select Category", list(PAPERS.keys()))
st.markdown("---")

# --------------------
# Custom Paper
# --------------------
if category == "Others (Custom)":
    custom_paper = st.text_input(
        "✍️ Enter research paper name",
        placeholder="e.g. Diffusion Models Beat GANs on Image Synthesis"
    )

    if st.button("Explain"):
        with st.spinner("Generating explanation..."):
            explanation = chain.invoke({
                "paper": custom_paper,
                "words": words,
                "depth": depth
            })

        st.markdown('<div class="explain-box">', unsafe_allow_html=True)
        st.write(explanation)
        st.markdown('</div>', unsafe_allow_html=True)

# --------------------
# Preset Papers
# --------------------
else:
    cols = st.columns(2)

    for i, paper in enumerate(PAPERS[category]):
        with cols[i % 2]:
            if st.button(paper, use_container_width=True):
                with st.spinner("Generating explanation..."):
                    explanation = chain.invoke({
                        "paper": paper,
                        "words": words,
                        "depth": depth
                    })

                st.markdown('<div class="explain-box">', unsafe_allow_html=True)
                st.write(explanation)
                st.markdown('</div>', unsafe_allow_html=True)
