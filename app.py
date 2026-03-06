"""
╔══════════════════════════════════════════════════════════════════╗
║  app.py  —  DataMind Platform                                    ║
║  THE DIRECTOR                                                    ║
║                                                                  ║
║  This is the MAIN file — the one you run to start the app.      ║
║  It ties everything together: the sidebar, the 5 pages,         ║
║  the file upload, the AI chat, and all the engines.             ║
║                                                                  ║
║  Think of it as the film director: they don't act, write the    ║
║  script, or build the sets themselves — they call on each       ║
║  specialist at the right time and stitch it all together.       ║
║                                                                  ║
║  Run with:  streamlit run app.py                                ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ── IMPORTS ──────────────────────────────────────────────────────────────────

import streamlit as st
# Streamlit is the magic library that turns Python scripts into web apps.
# Every st.button(), st.text_input(), st.plotly_chart() draws something on screen.
# When the user clicks anything, Python runs top-to-bottom again.

import pandas as pd     # DataFrames — our in-memory spreadsheet
import numpy as np      # Maths arrays
import io               # Input/Output: reading bytes in memory (for file handling)
import os               # OS tools: file paths, environment variables
import sys              # System tools: Python path manipulation
import tempfile         # Creates temporary files on disk (auto-deleted afterward)
import time             # For pausing or timing
import json             # JSON parsing and writing
from typing import Optional   # Type hint: "this might be None"

# ── Path setup ────────────────────────────────────────────────────────────────

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Adds the folder containing app.py to the Python search path.
# Without this, "from core.ollama_client import ..." might fail.
# __file__      = path to app.py itself
# os.path.abspath() = convert to absolute path
# os.path.dirname() = get the containing folder
# sys.path.insert(0, ...) = add it at position 0 (highest priority)

# ── Internal modules ──────────────────────────────────────────────────────────
# Import from OUR files inside the datamind project.

from ui.styles import (
    PARALLAX_CSS, hero_html, section_divider, metric_card_html,
    status_badge, sidebar_logo
)
# PARALLAX_CSS    : the big CSS string that styles the whole app (dark theme etc.)
# hero_html()     : returns the HTML for the animated landing page hero section
# section_divider : creates a horizontal divider with a label
# metric_card_html: creates a styled "stats card" widget (e.g. "891 Rows")
# status_badge    : green/red badge showing Ollama online/offline
# sidebar_logo    : the DataMind logo HTML for the sidebar

from core.ollama_client import get_client
# get_client() returns the singleton OllamaClient (see ollama_client.py).

from core.eda_engine import EDAEngine
# The data detective class (see eda_engine.py).

from core.automl_engine import AutoMLEngine
# The robot trainer class (see automl_engine.py).

from core.rag_engine import RAGEngine, extract_text_from_file
# RAGEngine       : the smart library pipeline
# extract_text_from_file : reads text from PDF/TXT/DOCX files

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
#  Must be called FIRST before any other Streamlit command.
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="DataMind AI Platform",   # Text shown on the browser tab
    page_icon="⬡",                       # Icon shown on the browser tab
    layout="wide",                       # "wide" = full browser width (no narrow column)
    initial_sidebar_state="expanded",    # Show the sidebar open on first load
)

st.markdown(PARALLAX_CSS, unsafe_allow_html=True)
# Inject the custom CSS into the page.
# unsafe_allow_html=True : allow raw HTML/CSS (Streamlit is cautious about this by default).
# Without this, HTML tags would be shown as plain text.


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE — The App's Persistent Memory
#
#  KEY CONCEPT: Streamlit re-runs this ENTIRE script from top to bottom
#  every time the user clicks a button, types something, or changes a widget.
#
#  Without session_state, all variables would reset on every click.
#  Session state is like a sticky note that survives reruns.
# ══════════════════════════════════════════════════════════════════════════════

def init_state():
    """Sets default values for all session state variables on first run."""
    defaults = {
        "df":            None,   # The uploaded Pandas DataFrame (None = nothing uploaded yet)
        "file_name":     None,   # Name of the uploaded file (used to detect NEW uploads)
        "eda_engine":    None,   # EDAEngine object (created when data is loaded)
        "automl_engine": None,   # AutoMLEngine object (created when training starts)
        "automl_results": None,  # List of model results dicts (filled after training)
        "rag_engine":    None,   # RAGEngine object (created when RAG Studio opens)
        "chat_history":  [],     # [(role, message), ...] for the Data Chat free-chat
        "rag_history":   [],     # [(role, message), ...] for the RAG Studio chat
        "nl_history":    [],     # [{"question":..., "code":..., "result":...}] for NL queries
        "selected_model": None,  # Which Ollama model is currently active
        "automl_done":   False,  # Has AutoML training been completed? True/False
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val
            # IMPORTANT: "if key not in" check!
            # If we wrote st.session_state[key] = val WITHOUT checking,
            # every single rerun would reset all values. That would break everything.
            # We only set the default if the key doesn't exist yet (first run).

init_state()
# Call it immediately so state is ready before anything else runs.


# ══════════════════════════════════════════════════════════════════════════════
#  OLLAMA CONNECTION
# ══════════════════════════════════════════════════════════════════════════════

client = get_client()
# Get the shared OllamaClient singleton.

ollama_online = client.is_online()
# Ping Ollama to see if it's running. Returns True or False.
# This runs on every rerun — so if you start Ollama, the badge updates automatically.

available_models = client.list_models() if ollama_online else []
# If Ollama is online, fetch the list of installed model names.
# If Ollama is offline, use an empty list (no models available).

if available_models and st.session_state.selected_model not in available_models:
    st.session_state.selected_model = available_models[0]
    # If we have models AND the currently selected model isn't in the list
    # (e.g. it was deleted or we're starting fresh), default to the first available model.


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — Always visible, regardless of which page is shown
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    # Everything inside "with st.sidebar:" appears in the left panel.

    st.markdown(sidebar_logo(), unsafe_allow_html=True)
    # Draw the DataMind ⬡ logo at the top of the sidebar.

    page = st.radio(
        "Navigation",
        ["⬡ Home", "🔬 EDA Lab", "🤖 AutoML Arena", "💬 Data Chat", "📚 RAG Studio"],
        label_visibility="collapsed"
        # label_visibility="collapsed" hides the "Navigation" label text.
        # The radio buttons show as clickable options.
    )
    # st.radio creates radio buttons. The selected value is stored in "page".
    # When the user clicks a different page, Streamlit reruns and "page" changes.

    st.markdown(section_divider("system"), unsafe_allow_html=True)
    # Draw a "SYSTEM" section divider line in the sidebar.

    # ── Ollama Status Badge ───────────────────────────────────────────────
    status_html = (status_badge("Ollama Online", online=True)
                   if ollama_online
                   else status_badge("Ollama Offline", online=False))
    st.markdown(status_html, unsafe_allow_html=True)
    # Show a green "Ollama Online" badge or red "Ollama Offline" badge.

    if ollama_online and available_models:
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        # A tiny spacer div (0.5rem = ~8 pixels of vertical space).

        selected_model = st.selectbox(
            "Active Model",
            options=available_models,   # The list of installed model names
            index=(available_models.index(st.session_state.selected_model)
                   if st.session_state.selected_model in available_models else 0),
            # index = which option to show as selected.
            # .index() finds the position of the current model in the list.
            # If it's not found (model was deleted), default to index 0.
            key="model_select"
            # key = unique identifier for this widget. Required when multiple widgets exist.
        )
        st.session_state.selected_model = selected_model
        # Store the selected model in session state so all pages can use it.

    elif not ollama_online:
        st.markdown("""
        <div style="font-family:'Space Mono',monospace;font-size:0.65rem;
                    color:rgba(255,23,68,0.8);padding:0.8rem;
                    border:1px solid rgba(255,23,68,0.2);border-radius:4px;
                    margin-top:0.5rem;line-height:1.8;">
          Run <code style="color:#00e5ff;">ollama serve</code> to start the LLM engine
        </div>
        """, unsafe_allow_html=True)
        # If Ollama is offline, show a helpful instruction instead of the model dropdown.

    st.markdown(section_divider("data"), unsafe_allow_html=True)

    # ── File Upload ───────────────────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "Dataset",
        type=["csv", "xlsx", "xls", "json", "parquet"],
        # Only allow these file types. Others will be rejected.
        label_visibility="collapsed"
    )
    # Returns a file-like object when a file is uploaded, or None if no file.

    if uploaded_file is not None:
        # A file has been dragged/dropped or selected.
        try:
            name = uploaded_file.name       # Get just the filename (e.g. "titanic.csv")
            ext = name.split(".")[-1].lower()  # Extract the extension → "csv"

            # Route to the correct pandas reader based on file type:
            if ext == "csv":
                df = pd.read_csv(uploaded_file)       # CSV: comma-separated values
            elif ext in ["xlsx", "xls"]:
                df = pd.read_excel(uploaded_file)     # Excel files
            elif ext == "json":
                df = pd.read_json(uploaded_file)      # JSON records
            elif ext == "parquet":
                df = pd.read_parquet(uploaded_file)   # Parquet: compressed columnar format
            else:
                df = None

            if df is not None and (st.session_state.file_name != name):
                # Only reinitialise everything if this is a DIFFERENT file.
                # If the same file is "uploaded" again (Streamlit reruns), skip reset.
                st.session_state.df            = df
                st.session_state.file_name     = name
                st.session_state.eda_engine    = EDAEngine(df)   # Pre-create the EDA engine
                st.session_state.automl_engine = None            # Reset AutoML state
                st.session_state.automl_results = None
                st.session_state.automl_done   = False
                st.session_state.chat_history  = []              # Clear old chat
                st.session_state.nl_history    = []

            if st.session_state.df is not None:
                df = st.session_state.df
                st.markdown(f"""
                <div style="font-family:'Space Mono',monospace;font-size:0.65rem;
                            color:rgba(0,229,255,0.7);margin-top:0.5rem;line-height:1.9;">
                  ✓ {name}<br/>
                  {df.shape[0]:,} rows · {df.shape[1]} cols
                </div>
                """, unsafe_allow_html=True)
                # Show a confirmation: ✓ titanic.csv · 891 rows · 12 cols

        except Exception as e:
            st.error(f"Parse error: {e}")
            # If the file can't be read (corrupt, wrong format), show an error.

    # ── Sample Data Buttons ───────────────────────────────────────────────
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    if st.button("▶ Load Sample (Titanic)", use_container_width=True):
        # use_container_width=True makes the button fill the full sidebar width.
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        try:
            df = pd.read_csv(url)
            # Download the CSV directly from GitHub (requires internet).
            # pd.read_csv() can read from a URL just like a local file.
            st.session_state.df            = df
            st.session_state.file_name     = "titanic.csv"
            st.session_state.eda_engine    = EDAEngine(df)
            st.session_state.automl_engine = None
            st.session_state.automl_results = None
            st.session_state.automl_done   = False
            st.session_state.chat_history  = []
            st.rerun()
            # st.rerun() forces Streamlit to re-run the entire script immediately.
            # This causes the app to refresh and show the new data.
        except Exception as e:
            st.error(f"Could not load sample: {e}")

    if st.button("▶ Load Sample (Iris)", use_container_width=True):
        from sklearn.datasets import load_iris
        # import sklearn's built-in Iris dataset (no internet needed — it's bundled)

        data = load_iris(as_frame=True)
        # as_frame=True → return as pandas DataFrames instead of numpy arrays

        df = data.frame
        # data.frame is a DataFrame with all 4 flower measurements + target number

        df["target_name"] = df["target"].map({0: "setosa", 1: "versicolor", 2: "virginica"})
        # .map() replaces numbers with their string names.
        # {0: "setosa", ...} is a mapping dictionary.
        # Now the DataFrame has a readable "target_name" column.

        st.session_state.df          = df
        st.session_state.file_name   = "iris.csv"
        st.session_state.eda_engine  = EDAEngine(df)
        st.session_state.automl_engine = None
        st.session_state.automl_results = None
        st.session_state.automl_done = False
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: HOME
#  The landing page with the animated parallax hero and feature cards.
# ══════════════════════════════════════════════════════════════════════════════

if page == "⬡ Home":
    # "if page ==" checks which radio button the user selected.
    # Only ONE of these if/elif blocks runs per rerun.

    st.markdown(hero_html(), unsafe_allow_html=True)
    # Call hero_html() to get the HTML string for the big animated header.
    # It includes the parallax layers, title, subtitle, and "begin" button.

    # Stats bar — four big numbers under the hero
    st.markdown("""
    <div style="background:rgba(8,15,26,0.9);...">
      <div>9+  ML Models</div>
      <div>5   AI Agents</div>
      <div>∞   Datasets</div>
      <div>$0  Cloud Cost</div>
    </div>
    """, unsafe_allow_html=True)
    # Raw HTML div for the stats bar. Styling in the style attribute.

    # Feature cards grid
    cols = st.columns(2)
    # Creates a 2-column layout. cols[0] is left, cols[1] is right.

    features = [
        ("🔬", "Automated EDA", "Upload any CSV and get instant...", "#00e5ff"),
        ("🤖", "AutoML Arena", "We train 9 ML models simultaneously...", "#7c4dff"),
        ("💬", "Natural Language Queries", "Ask questions in plain English...", "#ff6d00"),
        ("📚", "RAG Knowledge Studio", "Upload PDFs, research papers...", "#00e676"),
        ("⚡", "Streaming AI Responses", "All LLM responses stream token-by-token...", "#7c4dff"),
        ("🛡️", "100% Local & Private", "Your data never leaves your machine...", "#00e5ff"),
    ]
    # A list of tuples: (emoji, title, description, accent_colour)

    for i, (icon, title, desc, color) in enumerate(features):
        # enumerate() gives: (0, (🔬, EDA, ...)), (1, (🤖, AutoML, ...)), etc.
        with cols[i % 2]:
            # i % 2 = 0 for even i, 1 for odd i.
            # This alternates between left column and right column.
            # i=0 → cols[0] (left), i=1 → cols[1] (right), i=2 → cols[0] (left), etc.
            st.markdown(f"""
            <div style="background:rgba(13,27,46,0.7);...">
              <div>{icon}</div>
              <h3>{title}</h3>
              <p>{desc}</p>
              <div style="background:linear-gradient(90deg,{color},#7c4dff);"></div>
            </div>
            """, unsafe_allow_html=True)
            # Each card: icon + title + description + a coloured accent bottom border.
            # linear-gradient(90deg, color, #7c4dff) = gradient from feature colour to purple


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: EDA LAB
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔬 EDA Lab":
    # "elif" = "else if" — only runs if the previous "if" was False.

    # Page header HTML (MODULE 01, title, subtitle)
    st.markdown("""...<h1>Exploratory Data Analysis</h1>...""", unsafe_allow_html=True)

    df = st.session_state.df
    if df is None:
        st.info("⬡ Upload a dataset in the sidebar or load a sample to begin analysis.")
        st.stop()
        # st.stop() immediately halts execution of the rest of this page.
        # Like an early return — prevents the rest of the code from running on empty data.

    eda = st.session_state.eda_engine or EDAEngine(df)
    # Use existing engine OR create a new one.
    # "a or b" = use a if a is truthy, otherwise use b.
    # If eda_engine is None (falsy), create EDAEngine(df).

    st.session_state.eda_engine = eda
    # Store it back (in case we just created a new one above).

    # ── Metric Cards Row ─────────────────────────────────────────────────
    profile = eda.get_profile()

    col1, col2, col3, col4, col5 = st.columns(5)
    # Create 5 equal-width columns across the page.

    metric_data = [
        (col1, f"{profile['shape']['rows']:,}", "Rows"),
        (col2, profile['shape']['columns'], "Columns"),
        (col3, f"{profile['memory_mb']} MB", "Memory"),
        (col4, profile['duplicate_rows'], "Duplicates"),
        (col5, len(profile['missing']), "Cols with Nulls"),
    ]
    for col, val, label in metric_data:
        with col:
            st.markdown(metric_card_html(val, label), unsafe_allow_html=True)
            # Draw one stats card per column.

    # ── Data Preview ──────────────────────────────────────────────────────
    with st.expander("📋 Data Preview", expanded=False):
        # st.expander creates a collapsible section.
        # expanded=False means it starts collapsed (closed).
        st.dataframe(df.head(50), use_container_width=True, height=300)
        # Show the first 50 rows of the DataFrame.
        # use_container_width=True makes it fill the full page width.

    # ── Descriptive Statistics ────────────────────────────────────────────
    with st.expander("📊 Descriptive Statistics", expanded=True):
        # expanded=True means this section starts open.
        stats = eda.get_descriptive_stats()
        if not stats.empty:
            st.dataframe(stats, use_container_width=True)
        else:
            st.info("No numeric columns found.")

    # ── Visualisation Tabs ────────────────────────────────────────────────
    tabs = st.tabs([
        "📈 Distributions", "🔥 Correlations", "❓ Missing Values",
        "🏷️ Categories", "📦 Outliers", "🎯 Target Analysis"
    ])
    # st.tabs creates a row of tabs. tabs[0] is the first tab, etc.
    # Each "with tabs[n]:" block fills that tab's content.

    with tabs[0]:   # Distributions
        fig = eda.plot_distributions()
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            # st.plotly_chart renders a Plotly figure as an interactive chart.
        else:
            st.info("No numeric columns to plot.")

    with tabs[1]:   # Correlations
        fig = eda.plot_correlation_heatmap()
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need at least 2 numeric columns.")

    with tabs[2]:   # Missing Values
        fig = eda.plot_missing_values()
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("**Missing Value Detail:**")
            missing_data = pd.DataFrame([
                {"Column": col, "Count": v["count"], "Percent": f"{v['percent']}%"}
                for col, v in profile["missing"].items()
            ])
            # List comprehension building a list of row-dicts → DataFrame.
            # One row per column that has missing values.
            st.dataframe(missing_data, use_container_width=True)
        else:
            st.success("✓ No missing values found in this dataset!")

    with tabs[3]:   # Categorical counts
        fig = eda.plot_categorical_counts()
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No categorical columns found.")

    with tabs[4]:   # Outliers
        outlier_df = eda.detect_outliers_iqr()
        if not outlier_df.empty:
            fig = eda.plot_outlier_analysis()
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(outlier_df, use_container_width=True)
        else:
            st.info("No numeric columns for outlier analysis.")

    with tabs[5]:   # Target Analysis
        target_options = ["(Select target column)"] + df.columns.tolist()
        # The first option is a placeholder "please choose" option.
        # df.columns.tolist() adds all actual column names after it.

        target = st.selectbox("Target Column", target_options)
        if target != "(Select target column)":
            # Only proceed if user has made a real selection.
            fig = eda.plot_target_analysis(target)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

            # Custom scatter plot explorer
            num_cols = eda.numeric_cols
            if len(num_cols) >= 2:
                c1, c2, c3 = st.columns(3)
                x_col   = c1.selectbox("X Axis", num_cols, key="scatter_x")
                y_col   = c2.selectbox("Y Axis", num_cols, index=1, key="scatter_y")
                # index=1 means default to the SECOND column (index 0 is the first).
                color_col = c3.selectbox("Color", ["(none)"] + df.columns.tolist(), key="scatter_c")
                color_arg = None if color_col == "(none)" else color_col
                # If "(none)" was selected, pass None to the scatter function.
                # Otherwise pass the actual column name.
                fig = eda.plot_scatter(x_col, y_col, color_arg)
                st.plotly_chart(fig, use_container_width=True)

    # ── AI EDA Narrative ──────────────────────────────────────────────────
    if st.button("🧠 Generate AI EDA Insights", use_container_width=True):
        if not ollama_online:
            st.error("Ollama is offline. Start it with: ollama serve")
        else:
            summary = eda.get_summary_for_llm()
            # Build the text briefing document for the AI.

            stats_json = (eda.get_descriptive_stats().to_json()
                         if not eda.get_descriptive_stats().empty else "{}")
            # .to_json() converts the DataFrame to a JSON string.
            # If the DataFrame is empty, use "{}" (empty JSON object).

            output_container = st.empty()
            # st.empty() creates a placeholder that we can overwrite.
            # We'll overwrite it with more and more text as tokens stream in.

            full_text = ""   # Accumulate all tokens here.

            with st.spinner("Analyzing dataset..."):
                # st.spinner shows a spinning indicator while the block runs.
                for token in client.stream(
                    prompt=f"Dataset Summary:\n{summary}\n\nStatistics:\n{stats_json}\n\nGenerate 7 key data science insights.",
                    model=st.session_state.selected_model,
                    system="You are a senior data scientist. Provide numbered, specific, actionable insights.",
                ):
                    full_text += token   # Add new token to the growing string.
                    output_container.markdown(
                        f'<div class="terminal-box">{full_text}▌</div>',
                        unsafe_allow_html=True
                    )
                    # Overwrite the placeholder with the GROWING text + cursor (▌).
                    # This creates the "typewriter" streaming effect.

            output_container.markdown(
                f'<div class="terminal-box">{full_text}</div>',
                unsafe_allow_html=True
            )
            # Final update: remove the cursor (▌) once streaming is done.


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: AUTOML ARENA
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🤖 AutoML Arena":

    st.markdown("""...<h1>AutoML Arena</h1>...""", unsafe_allow_html=True)

    df = st.session_state.df
    if df is None:
        st.info("⬡ Upload a dataset in the sidebar first.")
        st.stop()

    # ── Configuration Row ─────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        target_col = st.selectbox("Target Column", df.columns.tolist())
        # Let user pick which column they want to predict.

    with col2:
        test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05,
                              help="Fraction of data held out for final evaluation")
        # st.slider(label, min, max, default, step, help=...)
        # Produces a slider from 0.1 to 0.4, default 0.2, step 0.05.
        # So the user can choose 10%, 15%, 20%, 25%, 30%, 35%, or 40%.

    with col3:
        # Show the detected problem type (classification or regression)
        if st.session_state.automl_engine:
            detected_type = st.session_state.automl_engine.problem_type
        else:
            # Quick preview before training starts:
            target_data = df[target_col].dropna()
            detected_type = ("classification"
                             if target_data.dtype in ["object", "category"] or target_data.nunique() <= 20
                             else "regression")
        st.markdown(f"""
        <div>Problem Type: <b>{detected_type.upper()}</b></div>
        """, unsafe_allow_html=True)

    # ── Fast Mode toggle ──────────────────────────────────────────────────
    fast_mode = st.checkbox(
        "⚡ Fast Mode — fewer trees, ~5x faster. Recommended for datasets > 1,000 rows.",
        value=len(df) > 1000,
        # Auto-tick Fast Mode for large datasets so training doesn't take forever.
    )

    # ── Launch Button ─────────────────────────────────────────────────────
    if st.button("🚀 Launch AutoML Training", use_container_width=True):
        st.session_state.automl_engine = AutoMLEngine(df, target_col, test_size, fast_mode=fast_mode)
        # Pass fast_mode flag — AutoML will use lighter models if True.

        st.session_state.automl_done = False
        # Reset the "done" flag so the results section hides while training.

        progress_bar = st.progress(0)
        # st.progress(0) shows a progress bar starting at 0%.

        status_text = st.empty()
        # Placeholder for the "Training Random Forest..." status message.

        def update_progress(val, msg):
            """Callback function called by AutoMLEngine after each model trains."""
            progress_bar.progress(val)
            # Update the bar. val is 0.0 to 1.0 (0% to 100%).
            status_text.markdown(f'<div style="...color:#00e5ff;">{msg}</div>',
                                 unsafe_allow_html=True)
            # Update the text message.

        with st.spinner("Training models..."):
            results = st.session_state.automl_engine.run(progress_callback=update_progress)
            # Run all 9 models. After each model, update_progress() is called.
            # "progress_callback=update_progress" passes our function as an argument.
            # AutoMLEngine calls it internally. This is the "callback pattern".

            st.session_state.automl_results = results
            st.session_state.automl_done = True
            # Store results and mark training as complete.

        st.rerun()
        # Rerun the script so the results section appears.

    # ── Results Section (only shows after training) ───────────────────────
    if st.session_state.automl_done and st.session_state.automl_engine:
        engine  = st.session_state.automl_engine
        results = st.session_state.automl_results

        if results:
            best    = results[0]   # Best model is first (sorted by metric)
            primary = "f1_weighted" if engine.problem_type == "classification" else "r2"

            # Winner banner
            st.markdown(f"""
            <div>
              Champion: {best['model_name']} |
              {primary}: {best.get(primary)} |
              CV: {best['cv_mean']} ±{best['cv_std']}
            </div>
            """, unsafe_allow_html=True)

        result_tabs = st.tabs(["🏆 Leaderboard", "📊 Charts", "📌 Feature Importance", "🤖 AI Explanation"])

        with result_tabs[0]:   # Leaderboard table
            leaderboard_df = engine.get_leaderboard_df()
            st.dataframe(leaderboard_df, use_container_width=True)

        with result_tabs[1]:   # Leaderboard bar + radar chart
            c1, c2 = st.columns(2)
            with c1:
                fig = engine.plot_leaderboard()
                if fig: st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = engine.plot_metrics_radar()
                if fig: st.plotly_chart(fig, use_container_width=True)

        with result_tabs[2]:   # Feature importance chart
            fig = engine.plot_feature_importance()
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                fi_df = engine.get_feature_importance()
                if fi_df is not None:
                    st.dataframe(fi_df[["feature", "importance_normalized"]].head(20),
                                 use_container_width=True)
            else:
                st.info("Feature importance not available for this model type.")

        with result_tabs[3]:   # AI model explanation
            if st.button("🧠 Generate AI Model Explanation"):
                if not ollama_online:
                    st.error("Ollama offline.")
                else:
                    best_result = results[0]
                    metrics = {k: v for k, v in best_result.items()
                               if k not in ["pipeline", "model_name"]}
                    # Dict comprehension: copy best_result but EXCLUDE the pipeline object
                    # (can't send a trained sklearn pipeline to the AI as text!)
                    # Also exclude model_name (not a metric).

                    fi_df   = engine.get_feature_importance()
                    fi_dict = (fi_df.set_index("feature")["importance_normalized"].head(10).to_dict()
                               if fi_df is not None else {})
                    # .set_index("feature") makes feature names the index.
                    # ["importance_normalized"] selects just that column.
                    # .head(10) takes top 10. .to_dict() converts to {feature: importance} dict.

                    explanation = client.explain_model(
                        best_result["model_name"], metrics, fi_dict,
                        model_llm=st.session_state.selected_model
                    )
                    st.markdown(f'<div class="terminal-box">{explanation}</div>',
                                unsafe_allow_html=True)

                    # Feature engineering suggestions
                    if st.session_state.eda_engine:
                        suggestions = client.suggest_features(
                            st.session_state.eda_engine.get_summary_for_llm(),
                            target_col,
                            model=st.session_state.selected_model
                        )
                        st.markdown(f'<div class="terminal-box">{suggestions}</div>',
                                    unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: DATA CHAT (Natural Language Queries)
# ══════════════════════════════════════════════════════════════════════════════

elif page == "💬 Data Chat":

    st.markdown("""...<h1>Data Chat</h1>...""", unsafe_allow_html=True)

    df = st.session_state.df
    if df is None:
        st.info("⬡ Upload a dataset first.")
        st.stop()

    eda = st.session_state.eda_engine or EDAEngine(df)

    chat_tabs = st.tabs(["🔍 NL Data Query", "🤖 Free Chat with Data Analyst"])

    # ── Tab 1: NL Query ───────────────────────────────────────────────────
    with chat_tabs[0]:
        # Shortcut buttons for example queries
        examples = [
            "What is the average age grouped by survival?",
            "Show me the top 5 most expensive rows",
            "How many missing values per column?",
            "What is the correlation between Age and Fare?",
        ]
        selected_example = None
        ex_cols = st.columns(len(examples))
        for i, ex in enumerate(examples):
            with ex_cols[i]:
                if st.button(f'"{ex[:30]}..."', key=f"ex_{i}", use_container_width=True):
                    selected_example = ex
                    # If this button was clicked, store the example text.
                    # It'll appear pre-filled in the text input below.

        nl_query = st.text_input(
            "Your Question",
            value=selected_example or "",
            # value: pre-populate with the selected example, or empty string.
            placeholder="e.g., What's the average salary by department?",
            key="nl_input"
        )

        if st.button("⬡ Execute Query", use_container_width=True) and nl_query:
            # "and nl_query" — only proceed if the input is not empty.
            if not ollama_online:
                st.error("Ollama offline.")
            else:
                # Build a mini description of the DataFrame for the AI
                df_info = f"""Columns: {df.columns.tolist()}
Dtypes: {df.dtypes.to_dict()}
Shape: {df.shape}
Sample values:
{df.head(3).to_string()}"""
                # df.head(3).to_string() converts the first 3 rows to plain text.
                # This gives the AI a concrete example of what the data looks like.

                with st.spinner("Translating to Pandas..."):
                    code = client.nl_to_pandas(df_info, nl_query,
                                               model=st.session_state.selected_model)
                    # Send the question + schema to the NL-to-Pandas agent.
                    # Returns a string of Python code.

                # Clean up code fences the AI might have added
                code = code.strip().replace("```python", "").replace("```", "").strip()
                # The AI sometimes wraps code in markdown fences: ```python ... ```
                # We strip those so exec() can run clean code.

                st.markdown("**Generated Code:**")
                st.code(code, language="python")
                # st.code() renders the code with syntax highlighting (prettily).

                # ── Execute the AI-generated code ──────────────────────────
                try:
                    local_vars = {"df": df.copy(), "pd": pd, "np": np}
                    # Create a "sandboxed" namespace for exec().
                    # The code can see df, pd, np but NOTHING ELSE.
                    # This limits potential damage from bad AI-generated code.

                    exec(code, {}, local_vars)
                    # exec(code, globals, locals):
                    # code       : the string of Python code to run
                    # {} (global): empty dict = no access to global modules/builtins
                    # local_vars : the variables the code can read/write

                    result = local_vars.get("result", None)
                    # The AI was instructed to store its answer in a variable named "result".
                    # We read it back from local_vars after execution.

                    if result is not None:
                        st.markdown("**Result:**")
                        if isinstance(result, pd.DataFrame):
                            st.dataframe(result, use_container_width=True)
                            # isinstance() checks if "result" IS a DataFrame.
                        elif isinstance(result, pd.Series):
                            st.dataframe(result.to_frame(), use_container_width=True)
                            # .to_frame() converts a Series to a single-column DataFrame.
                        else:
                            st.markdown(f'<div class="terminal-box">{result}</div>',
                                        unsafe_allow_html=True)
                            # For any other type (number, string, etc.) show in a terminal box.

                        st.session_state.nl_history.append({
                            "question": nl_query,
                            "code": code,
                            "result": str(result)[:200]  # Truncate long results
                        })
                    else:
                        st.warning("Code executed but no `result` variable was set.")

                except Exception as e:
                    st.error(f"Execution error: {e}")
                    st.info("Try rephrasing your question or check column names.")

        # Query history expander
        if st.session_state.nl_history:
            with st.expander(f"📜 Query History ({len(st.session_state.nl_history)} queries)"):
                for item in reversed(st.session_state.nl_history[-10:]):
                    # reversed() shows newest first.
                    # [-10:] shows only the last 10 items.
                    st.markdown(f"**Q:** {item['question']}")
                    st.code(item['code'], language="python")
                    st.markdown(f"**Result:** `{item['result']}`")
                    st.markdown("---")
                    # "---" creates a horizontal divider line in markdown.

    # ── Tab 2: Free Chat ──────────────────────────────────────────────────
    with chat_tabs[1]:
        # Display existing conversation
        chat_container = st.container()
        with chat_container:
            if not st.session_state.chat_history:
                st.markdown("""<div>Start a conversation with your AI data analyst.</div>""",
                           unsafe_allow_html=True)
            else:
                for role, msg in st.session_state.chat_history:
                    # Loop over every message in the chat history.
                    # role = "user" or "assistant"
                    if role == "user":
                        st.markdown(f'<div class="msg-user"><b>YOU</b><br/>{msg}</div>',
                                   unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="msg-ai"><b>DATAMIND AI</b><br/>{msg}</div>',
                                   unsafe_allow_html=True)

        # Message input
        col1, col2 = st.columns([5, 1])
        with col1:
            user_msg = st.text_input("Message", placeholder="Ask anything about your data...",
                                     key="chat_input", label_visibility="collapsed")
        with col2:
            send = st.button("Send ⬡", use_container_width=True)

        if send and user_msg:
            if not ollama_online:
                st.error("Ollama offline.")
            else:
                st.session_state.chat_history.append(("user", user_msg))
                # Add the user's message to history immediately.

                messages = [
                    {"role": role, "content": msg}
                    for role, msg in st.session_state.chat_history
                ]
                # Convert our [(role, msg), ...] format to Ollama's format:
                # [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

                system = f"""You are DataMind, an expert AI data scientist.
You have access to this dataset:
{eda.get_summary_for_llm()}

Help the user understand their data. Be specific, reference actual column names."""
                # System prompt includes a summary of the ACTUAL dataset.
                # This way the AI can reference real column names and statistics.

                response_container = st.empty()   # Placeholder for streaming output
                full_response = ""

                for token in client.chat(messages=messages,
                                         model=st.session_state.selected_model,
                                         system=system):
                    full_response += token
                    response_container.markdown(
                        f'<div class="msg-ai">DATAMIND AI<br/>{full_response}▌</div>',
                        unsafe_allow_html=True
                    )
                    # Overwrite the placeholder every token — typewriter effect.

                response_container.empty()
                # Clear the placeholder (the permanent version is added below).

                st.session_state.chat_history.append(("assistant", full_response))
                # Add the completed AI response to history.

                st.rerun()
                # Rerun so the message appears in the chat_container above.

        if st.session_state.chat_history:
            if st.button("🗑 Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: RAG STUDIO
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📚 RAG Studio":

    st.markdown("""...<h1>RAG Knowledge Studio</h1>...""", unsafe_allow_html=True)

    # Initialise RAG engine if it doesn't exist yet
    if st.session_state.rag_engine is None:
        st.session_state.rag_engine = RAGEngine()
        # Create a fresh RAG engine the FIRST time this page is visited.
        # On subsequent visits, session state preserves the existing engine (and its data).

    rag: RAGEngine = st.session_state.rag_engine
    # Type annotation (: RAGEngine) helps IDEs and readers understand what this variable is.
    # It doesn't change any runtime behaviour — just documentation.

    status = rag.get_status()
    # Get current stats: how many chunks, how many files, what sources.

    # Stats cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(metric_card_html(status["total_chunks"], "Indexed Chunks", color="#00e676"),
                   unsafe_allow_html=True)
    with col2:
        st.markdown(metric_card_html(status["files_ingested"], "Files Ingested", color="#00e676"),
                   unsafe_allow_html=True)
    with col3:
        st.markdown(metric_card_html(len(status["sources"]), "Sources", color="#00e676"),
                   unsafe_allow_html=True)

    rag_tabs = st.tabs(["📁 Document Ingestion", "💬 Ask Knowledge Base", "⚙️ RAG Settings"])

    # ── Tab 1: Ingestion ──────────────────────────────────────────────────
    with rag_tabs[0]:
        col1, col2 = st.columns([3, 2])

        with col1:
            doc_files = st.file_uploader(
                "Upload Documents",
                type=["txt", "pdf", "md", "py", "json", "csv", "docx"],
                accept_multiple_files=True,
                # accept_multiple_files=True allows uploading more than one file at once.
                label_visibility="collapsed"
            )

            chunk_size = st.slider("Chunk Size (characters)", 200, 1000, 600, 50)
            overlap    = st.slider("Chunk Overlap", 0, 200, 100, 25)

            if st.button("⬡ Ingest Documents", use_container_width=True) and doc_files:
                for doc_file in doc_files:
                    # Loop over every uploaded file.

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=f".{doc_file.name.split('.')[-1]}"
                        # Give the temp file the same extension as the original.
                    ) as tmp:
                        tmp.write(doc_file.read())
                        # Read bytes from the Streamlit upload and write to a real disk file.
                        # Streamlit file objects live in memory. rag_engine needs a file path.
                        tmp_path = tmp.name   # The path to the temp file on disk.

                    with st.spinner(f"Ingesting {doc_file.name}..."):
                        result = rag.ingest_file(tmp_path, doc_file.name,
                                                 chunk_size=chunk_size, overlap=overlap)

                    os.unlink(tmp_path)
                    # Delete the temporary file after we're done with it.
                    # "unlink" = delete in Unix. Like os.remove().

                    if result["status"] == "success":
                        st.success(f"✓ {doc_file.name}: {result['chunks']} chunks indexed")
                    else:
                        st.error(f"✗ {doc_file.name}: {result.get('message', 'Error')}")

                st.rerun()   # Refresh stats cards

        with col2:
            # Info panel explaining how RAG works
            st.markdown("""<div>How It Works: chunk → embed → store → retrieve → answer</div>""",
                       unsafe_allow_html=True)

            df = st.session_state.df
            if df is not None:
                if st.button("⬡ Index Current Dataset as Context", use_container_width=True):
                    eda = st.session_state.eda_engine or EDAEngine(df)
                    text = eda.get_summary_for_llm()
                    # Convert the loaded DataFrame to a text summary.

                    result = rag.ingest_text(text,
                                             source_name=f"dataset:{st.session_state.file_name}")
                    # Add the dataset summary to the knowledge base as text.
                    # Now you can ask questions that combine document knowledge + dataset stats.

                    st.success(f"✓ Dataset indexed: {result['chunks']} chunks")
                    st.rerun()

        # List of ingested files
        if status["files"]:
            for f in status["files"]:
                st.markdown(f"""
                <div>📄 {f['file_name']} → {f['chunks']} chunks · {f['total_chars']:,} chars</div>
                """, unsafe_allow_html=True)

            if st.button("🗑 Clear Knowledge Base"):
                rag.clear()                          # Wipe all vectors and text
                st.session_state.rag_history = []    # Clear the chat history too
                st.rerun()

    # ── Tab 2: Q&A with Knowledge Base ────────────────────────────────────
    with rag_tabs[1]:
        if status["total_chunks"] == 0:
            st.info("⬡ Ingest documents first to start asking questions.")
        else:
            # Show existing conversation
            for role, msg in st.session_state.rag_history:
                css_class = "msg-user" if role == "user" else "msg-ai"
                label     = "YOU" if role == "user" else "DATAMIND RAG"
                st.markdown(f'<div class="{css_class}">{label}<br/>{msg}</div>',
                           unsafe_allow_html=True)

            top_k = st.slider("Context Chunks to Retrieve", 2, 10, 5, key="rag_topk")
            # How many similar chunks to pull from the vector store.
            # More chunks = more context for the AI, but longer prompt.

            rag_question = st.text_input("Ask your knowledge base",
                                          placeholder="What does the paper say about...?",
                                          key="rag_q")

            col1, col2 = st.columns([3, 1])
            with col1:
                ask = st.button("⬡ Ask RAG", use_container_width=True)
            with col2:
                show_context = st.checkbox("Show Context", value=False)
                # Checkbox: if checked, show which document chunks were retrieved.

            if ask and rag_question:
                if not ollama_online:
                    st.error("Ollama offline.")
                else:
                    prompt, retrieved = rag.build_prompt(rag_question, top_k=top_k)
                    # build_prompt returns BOTH the full augmented prompt AND
                    # the list of retrieved chunks (for display if show_context=True).

                    if show_context:
                        with st.expander("📋 Retrieved Context Chunks"):
                            for chunk in retrieved:
                                st.markdown(f"""
                                Source: {chunk['metadata'].get('source','?')} | Score: {chunk['score']}<br/>
                                {chunk['text'][:300]}...
                                """, unsafe_allow_html=True)

                    st.session_state.rag_history.append(("user", rag_question))
                    response_container = st.empty()
                    full_answer = ""

                    for token in client.stream(
                        prompt=prompt,              # The augmented prompt (context + question)
                        model=st.session_state.selected_model,
                        temperature=0.3,            # Lower temperature → more factual/grounded answers
                    ):
                        full_answer += token
                        response_container.markdown(
                            f'<div class="msg-ai">DATAMIND RAG<br/>{full_answer}▌</div>',
                            unsafe_allow_html=True
                        )

                    response_container.empty()
                    st.session_state.rag_history.append(("assistant", full_answer))
                    st.rerun()

    # ── Tab 3: Settings / How RAG Works ───────────────────────────────────
    with rag_tabs[2]:
        st.markdown("""
        <div>
          Embedding: Character n-gram hashing (128-dim). Upgrade to sentence-transformers for better semantics.<br/>
          Vector Store: In-memory cosine similarity. Swap with ChromaDB / FAISS for production.<br/>
          Chunking: Overlapping character windows. Overlap prevents boundary sentence loss.<br/>
          Generation: Context + query fused into structured prompt → Ollama generates grounded answer.
        </div>
        """, unsafe_allow_html=True)
        # This informational panel explains the RAG architecture to users.
        # It also serves as upgrade guidance for developers.
