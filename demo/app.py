"""
EasyRead Streamlit Web Application
Research Showcase for "Automated Text Simplification for Easy Read Generation"
"""

import os
import json
from pathlib import Path
import streamlit as st

# Page Configuration - Centered & Clean
st.set_page_config(
    page_title="EasyRead — Automated Text Simplification",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Minimal Theme-Native Typography
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Newsreader', Georgia, serif;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Newsreader', Georgia, serif !important;
    }

    /* Tab Headings: Serif */
    button[data-baseweb="tab"], 
    .stTabs [data-baseweb="tab"] p, 
    div[data-baseweb="tab-list"] button {
        font-family: 'Newsreader', Georgia, serif !important;
        font-size: 1.08rem !important;
        font-weight: 500 !important;
    }

    /* Dropdowns / Selectbox: Sans Serif */
    div[data-testid="stSelectbox"] *, 
    div[data-baseweb="select"] *, 
    div[role="listbox"] *,
    ul[role="listbox"] li {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 3rem;
        max-width: 780px;
    }

    .header-title {
        font-family: 'Newsreader', Georgia, serif;
        font-size: 2.5rem;
        font-weight: 500;
        letter-spacing: -0.02em;
        text-align: center;
        margin-bottom: 0.3rem;
    }
    
    .header-subtitle {
        font-family: 'Newsreader', Georgia, serif;
        font-style: italic;
        font-size: 1.15rem;
        opacity: 0.75;
        text-align: center;
        margin-bottom: 0.8rem;
    }

    .header-links {
        text-align: center;
        font-family: 'Inter', sans-serif;
        font-size: 0.88rem;
        margin-bottom: 2rem;
    }

    .quote-text {
        font-family: 'Newsreader', Georgia, serif;
        font-size: 1.12rem;
        line-height: 1.6;
        padding: 0.8rem 1rem;
        border-left: 3px solid #94A3B8;
        margin-bottom: 1.5rem;
        opacity: 0.9;
    }
    
    .output-text {
        font-family: 'Newsreader', Georgia, serif;
        font-size: 1.08rem;
        line-height: 1.65;
    }

    .concept-chip {
        display: inline-block;
        background-color: rgba(148, 163, 184, 0.12);
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 6px;
        padding: 4px 10px;
        font-family: 'Inter', sans-serif;
        font-size: 0.82rem;
        font-weight: 500;
        margin-right: 6px;
        margin-bottom: 6px;
    }

    .step-title {
        font-family: 'Newsreader', Georgia, serif;
        font-size: 1.08rem;
        font-weight: 500;
        margin-bottom: 0.3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_samples():
    samples_path = Path(__file__).parent / "samples.json"
    if samples_path.exists():
        with open(samples_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


samples = load_samples()

# Header - Title & Links Only
st.markdown('<div class="header-title">Automated Text Simplification for Easy Read Generation</div>', unsafe_allow_html=True)
st.markdown('<div class="header-subtitle">Graph-Guided Text Simplification for Easy Read Conversion</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="header-links">
        <a href="https://github.com/nominr/easyread" target="_blank">GitHub Repository</a> &nbsp;•&nbsp; 
        <a href="https://colab.research.google.com/drive/1r3FGknl7HhBJSu_tHqW72X96qLIRCKSE?usp=sharing" target="_blank">Research Colab Notebook</a>
    </div>
    """,
    unsafe_allow_html=True,
)

# Tabs
tab1, tab2 = st.tabs(["Benchmark Demonstrations", "Research Paper & Methodology"])

# TAB 1: BENCHMARK DEMONSTRATIONS
with tab1:
    if samples:
        categories = [f"{s['category']} — {s['title']}" for s in samples]
        selected_idx = st.selectbox(
            "Select Benchmark Example:",
            range(len(categories)),
            format_func=lambda i: categories[i],
            label_visibility="collapsed",
        )
        sample = samples[selected_idx]

        st.markdown(f'<div class="quote-text">"{sample["original_text"]}"</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.caption("STANDARD LLM BASELINE")
                st.markdown(f'<div class="output-text">{sample["baseline_output"]}</div>', unsafe_allow_html=True)
        with col2:
            with st.container(border=True):
                st.caption("GRAPH-GUIDED EASYREAD")
                st.markdown(f'<div class="output-text">{sample["graph_guided_output"].replace("\n", "<br>")}</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("View GNN & Knowledge Graph Analysis"):
            stats = sample["stats"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Concepts / Sent", stats["avg_nodes_per_sent"])
            c2.metric("Graph Depth", stats["max_path_length"])
            c3.metric("Coref Links", stats["num_coref_edges"])
            c4.metric("Graph Density", stats["graph_density"])

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Preserved Key Entities (Degree Centrality Ranked):**")
            chips = "".join(
                [f'<span class="concept-chip">{c[0]} ({c[1]:.2f})</span>' for c in sample["top_concepts"]]
            )
            st.markdown(chips, unsafe_allow_html=True)

# TAB 2: RESEARCH PAPER & METHODOLOGY
with tab2:
    st.markdown("### Abstract")
    st.write(
        """
        Easy Read is an accessibility format designed to make written information more understandable for people with intellectual disabilities, 
        but producing high-quality Easy Read documents requires trained specialists and does not scale easily to large collections of text. 
        Recent large language models (LLMs) offer a promising approach to automated simplification, yet naive prompting can drop important 
        concepts, weaken semantic faithfulness, or fail to follow the specific syntactic and structural constraints of Easy Read writing. 
        This paper presents a graph-guided LLM pipeline for Easy Read text simplification. We represent standard and Easy Read texts using 
        Abstract Meaning Representation (AMR) graphs, train a Graph Neural Network (GNN) classifier to learn structural differences between 
        the two forms, and convert graph-derived features into human-readable guidance for an LLM. A separate knowledge graph is used to rank 
        central concepts and encourage concept preservation during generation. We compare this graph-guided approach against a baseline LLM prompt 
        using SARI, BLEU, Flesch-Kincaid Grade Level, concept preservation, and average sentence length. Results show that graph-guided prompting 
        improves SARI, BLEU, and concept preservation relative to the baseline while maintaining comparable readability levels.
        """
    )

    st.markdown("---")
    st.markdown("### Methodology Pipeline")

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown('<div class="step-title">1. AMR Graph Construction</div>', unsafe_allow_html=True)
            st.caption("Parses text into sentence AMRs via fine-tuned BART-large (`model_parse_xfm_bart_large`). Individual graphs are merged into a document graph using semantic roles (:ARG0, :mod), discourse flow (NEXT_SENT), and coreference (COREF) edges.")

        with st.container(border=True):
            st.markdown('<div class="step-title">3. GNN Architecture & Training</div>', unsafe_allow_html=True)
            st.caption("Trains a Graph Convolutional Network (`EasyReadGNN`: 2-layer GCNConv + global mean pooling) to classify document graphs. Optimized with BCEWithLogitsLoss, Adam optimizer, and early stopping.")

    with c2:
        with st.container(border=True):
            st.markdown('<div class="step-title">2. Node Feature Embedding</div>', unsafe_allow_html=True)
            st.caption("Embeds AMR concept strings into 384-dimensional node feature vectors using `all-MiniLM-L6-v2` SentenceTransformer after stripping sense numbers.")

        with st.container(border=True):
            st.markdown('<div class="step-title">4. Knowledge Graph & Concept Ranking</div>', unsafe_allow_html=True)
            st.caption("Constructs a spaCy knowledge graph (`en_core_web_trf`), merging entity spans (e.g., '28 days'). Degree centrality ranks top concepts into 'must-use' and 'supporting' tiers.")

    st.markdown("---")
    st.markdown("### Evaluation Results (1,909 Parallel Corpus Records)")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("SARI Score", "15.009", "+4.853 vs Baseline")
    with col_b:
        st.metric("BLEU Score", "0.255", "+0.112 vs Baseline")
    with col_c:
        st.metric("Concept Preservation", "88.8%", "+25.1% vs Baseline")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Corpus-Level Evaluation Metrics:**")

    metrics_table = {
        "Metric": ["SARI (Primary)", "BLEU", "Concept Preservation", "FK Readability Grade", "Avg Sentence Length"],
        "Baseline Prompt": ["10.156 ± 5.162", "0.143 ± 0.143", "63.7% ± 26.7%", "6.745 ± 4.158", "6.271 ± 1.581"],
        "Graph-Guided System": ["15.009 ± 5.451", "0.255 ± 0.177", "88.8% ± 15.0%", "6.839 ± 3.999", "5.898 ± 1.502"],
        "Difference (Δ)": ["+4.853 (p < 0.001)", "+0.112 (p < 0.001)", "+25.1% (p < 0.001)", "+0.094", "-0.373 words"],
    }
    st.table(metrics_table)

    st.caption("Per-record win rate: Graph-guided output outperforms baseline on 81.0% of records (1,547 / 1,909 records).")

    st.markdown("---")
    st.markdown("### Paper Resources")
    st.markdown(
        """
        - **GitHub Repository**: [github.com/nominr/easyread](https://github.com/nominr/easyread)
        - **Colab Training Pipeline**: [Google Colab Research Notebook](https://colab.research.google.com/drive/1r3FGknl7HhBJSu_tHqW72X96qLIRCKSE?usp=sharing)
        - **Custom Input**: In development.
        """
    )
