"""
EasyRead Streamlit Web Application
Showcases Graph-Guided Text Simplification for Easy Read conversion.
"""

import os
import json
from pathlib import Path
import streamlit as st
from pipeline import EasyReadPipeline

# Page Configuration
st.set_page_config(
    page_title="EasyRead: Graph-Guided Text Simplification",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 12px 16px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #2563EB;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .concept-tag {
        display: inline-block;
        background-color: #EFF6FF;
        color: #1D4ED8;
        border: 1px solid #BFDBFE;
        border-radius: 16px;
        padding: 4px 12px;
        font-size: 0.85rem;
        font-weight: 500;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .baseline-box {
        background-color: #FFF5F5;
        border-left: 4px solid #E53E3E;
        padding: 16px;
        border-radius: 6px;
        font-size: 1.05rem;
    }
    .guided-box {
        background-color: #F0FDF4;
        border-left: 4px solid #16A34A;
        padding: 16px;
        border-radius: 6px;
        font-size: 1.05rem;
    }
    </style>
    """,
    unsafe_allow_code_html=True,
)


@st.cache_data
def load_samples():
    samples_path = Path(__file__).parent / "samples.json"
    if samples_path.exists():
        with open(samples_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


@st.cache_resource
def get_pipeline():
    model_dir = Path(__file__).parent.parent / "models" / "model_parse_xfm_bart_base-v0_1_0"
    return EasyReadPipeline(model_dir=str(model_dir) if model_dir.exists() else None)


# Load data & pipeline
samples = load_samples()
pipeline = get_pipeline()

# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/reading.png", width=64)
st.sidebar.title("EasyRead Demo")
st.sidebar.markdown(
    "**Graph Neural Network (GNN)** & **Abstract Meaning Representation (AMR)** guided simplification."
)

mode = st.sidebar.radio(
    "Select Mode",
    ["⚡ Quick Demo (Pre-computed)", "🧪 Live Pipeline Test"],
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔑 OpenAI API Key")
user_api_key = st.sidebar.text_input(
    "Enter OpenAI Key (optional)",
    type="password",
    help="Enter your key to run custom live LLM generation.",
)
if user_api_key:
    os.environ["OPENAI_API_KEY"] = user_api_key

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Paper Abstract Summary**:
    Standard LLMs often drop critical concepts or fail to simplify sentence structure adequately.
    Our approach extracts document-level AMR graphs, analyzes structural topology with a GNN,
    and ranks key entities via a SpaCy Knowledge Graph to construct structural prompts for LLMs.
    """
)

# Header Section
st.markdown('<div class="main-title">📖 EasyRead: Graph-Guided Text Simplification</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Empowering cognitive accessibility through AMR Graph Neural Networks & Knowledge Graph entity preservation.</div>',
    unsafe_allow_html=True,
)

# MAIN INTERFACE LOGIC
if mode == "⚡ Quick Demo (Pre-computed)":
    st.info("💡 **Quick Demo Mode**: Instantly test pre-computed real-world examples without waiting for AMR parsing or requiring an API key.")

    categories = [s["category"] + " — " + s["title"] for s in samples]
    selected_sample_idx = st.selectbox("Choose a sample text:", range(len(categories)), format_func=lambda i: categories[i])
    selected_sample = samples[selected_sample_idx]

    # Display original text
    st.subheader("📄 Original Complex Text")
    st.text_area("Complex Input", value=selected_sample["original_text"], height=100, disabled=True)

    # Display Graph & GNN Statistics
    st.subheader("📊 GNN & Structural Graph Analysis")
    stats = selected_sample["stats"]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{stats["avg_nodes_per_sent"]}</div><div class="metric-label">Avg Concepts/Sentence</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{stats["max_path_length"]}</div><div class="metric-label">Graph Concept Depth</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{stats["num_coref_edges"]}</div><div class="metric-label">Coref Entity Links</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{stats["graph_density"]}</div><div class="metric-label">Graph Density</div></div>',
            unsafe_allow_html=True,
        )

    # Key Concepts Badges
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**🧠 Knowledge Graph Extracted Key Entities (Must Preserve):**")
    concepts_html = "".join(
        [f'<span class="concept-tag">{c[0]} ({c[1]:.2f})</span>' for c in selected_sample["top_concepts"]]
    )
    st.markdown(concepts_html, unsafe_allow_html=True)

    st.markdown("---")

    # Side-by-Side Model Output Comparison
    st.subheader("⚔️ Model Output Comparison")
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### 🔴 Standard LLM Baseline Prompt")
        st.markdown(f'<div class="baseline-box">{selected_sample["baseline_output"]}</div>', unsafe_allow_html=True)
        st.caption("Standard prompt with general readability instructions.")

    with col_right:
        st.markdown("### 🟢 Our Graph-Guided EasyRead Pipeline")
        st.markdown(f'<div class="guided-box">{selected_sample["graph_guided_output"]}</div>', unsafe_allow_html=True)
        st.caption("GNN structural constraints + ranked concept entity protection.")

else:
    st.subheader("🧪 Live Custom Text Processing")
    custom_text = st.text_area(
        "Enter custom complex text to simplify:",
        value="The tenant shall have the right to terminate the tenancy agreement upon provision of written notice of not less than 28 days to the landlord.",
        height=120,
    )

    if st.button("🚀 Process & Generate Easy Read", type="primary"):
        with st.spinner("Analyzing document graph with AMR & SpaCy Knowledge Graph..."):
            doc_G, stats, ranked = pipeline.process_text(custom_text)
            baseline_prompt, graph_prompt, stats, ranked = pipeline.generate_prompts(custom_text)

        st.success("Graph analysis complete!")

        # Metrics display
        st.subheader("📊 Live GNN & Structural Graph Analysis")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                f'<div class="metric-card"><div class="metric-value">{stats["avg_nodes_per_sent"]}</div><div class="metric-label">Avg Concepts/Sentence</div></div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div class="metric-card"><div class="metric-value">{stats["max_path_length"]}</div><div class="metric-label">Graph Concept Depth</div></div>',
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f'<div class="metric-card"><div class="metric-value">{stats["num_coref_edges"]}</div><div class="metric-label">Coref Entity Links</div></div>',
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                f'<div class="metric-card"><div class="metric-value">{stats["graph_density"]}</div><div class="metric-label">Graph Density</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**🧠 Knowledge Graph Extracted Key Entities:**")
        concepts_html = "".join(
            [f'<span class="concept-tag">{c[0]} ({c[1]:.2f})</span>' for c in ranked[:8]]
        ) if ranked else "<span>No entities extracted</span>"
        st.markdown(concepts_html, unsafe_allow_html=True)

        st.markdown("---")

        # LLM Generation
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            with st.spinner("Invoking LLM for side-by-side generation..."):
                baseline_output = pipeline.run_llm(baseline_prompt, api_key=api_key)
                graph_output = pipeline.run_llm(graph_prompt, api_key=api_key)

            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown("### 🔴 Standard LLM Baseline Prompt")
                st.markdown(f'<div class="baseline-box">{baseline_output}</div>', unsafe_allow_html=True)
            with col_right:
                st.markdown("### 🟢 Our Graph-Guided EasyRead Pipeline")
                st.markdown(f'<div class="guided-box">{graph_output}</div>', unsafe_allow_html=True)
        else:
            st.warning("🔑 Please enter an OpenAI API key in the sidebar to generate live LLM responses.")

        with st.expander("🔍 Inspect Generated Prompts"):
            st.markdown("#### Baseline Prompt")
            st.code(baseline_prompt)
            st.markdown("#### Graph-Guided Prompt")
            st.code(graph_prompt)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #94A3B8; font-size: 0.85rem;'>"
    "EasyRead Research Project Demo • Powered by PyTorch Geometric, amrlib, SpaCy & Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
