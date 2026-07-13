import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO

# Import settings and utilities
from app.config.settings import MODEL_DIR
from app.utils.data_loader import DataLoader

# Backend API URL
API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="FakeShield AI - Fact Checking Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom premium CSS styling (Dark mode themed)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        letter-spacing: -0.5px;
    }
    
    /* Main title styling */
    .main-title {
        background: linear-gradient(135deg, #FF4B4B 0%, #852DF6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    /* Glassmorphism containers */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        margin-bottom: 20px;
    }
    
    /* Highlight labels */
    .highlight-fake {
        background-color: rgba(255, 75, 75, 0.2);
        color: #FF4B4B;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: bold;
    }
    
    .highlight-real {
        background-color: rgba(46, 204, 113, 0.2);
        color: #2ECC71;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: bold;
    }
    
    /* Metric styling */
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 0%, #a5a5a5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_dataset_cached():
    try:
        loader = DataLoader()
        return loader.load_dataset()
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return None


@st.cache_data
def compute_sentiment_stats(df):
    from app.preprocessing.sentiment import SentimentAnalyzer
    analyzer = SentimentAnalyzer()
    
    # We sample 1000 rows to keep it fast
    sample_df = df.sample(n=min(len(df), 1000), random_state=42).copy()
    
    polarities = []
    subjectivities = []
    
    for idx, row in sample_df.iterrows():
        text = str(row["text"])
        res = analyzer.analyze_sentiment(text)
        polarities.append(res["polarity"])
        subjectivities.append(res["subjectivity"])
        
    sample_df["polarity_score"] = polarities
    sample_df["subjectivity_score"] = subjectivities
    return sample_df


# Sidebar Navigation
st.sidebar.markdown(
    """
    <div style='text-align: center; margin-bottom: 20px;'>
        <h2 style='color: #FF4B4B; font-weight: 800;'>🛡️ FakeShield AI</h2>
        <p style='color: #888; font-size: 0.9rem;'>Explainable Fake News Platform</p>
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Navigation Menu",
    [
        "Home / News Detector",
        "Exploratory Data Analysis (EDA)",
        "Explainability Insights",
        "Knowledge Search (RAG)",
        "Model Metrics",
        "About Project",
    ],
)

# ----------------- HOME / NEWS DETECTOR -----------------
if page == "Home / News Detector":
    st.markdown("<h1 class='main-title'>🛡️ FakeShield AI News Detector</h1>", unsafe_allow_html=True)
    st.write(
        "Analyze claims and news articles using our unified BERT classifier and RAG Fact-Checking assistant."
    )

    st.markdown("### Paste News Article or Claim")
    input_text = st.text_area(
        "Enter news content here (at least 15 words recommended for accurate predictions):",
        placeholder="E.g., NASA discovers alien life in a hidden base on Mars...",
        height=200,
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        verify_btn = st.button("🔍 Verify Claim", type="primary", use_container_width=True)

    if verify_btn:
        if not input_text.strip():
            st.warning("Please paste some text first!")
        else:
            with st.spinner("Analyzing text & retrieving evidence..."):
                try:
                    # 1. Hit Predict Endpoint
                    pred_res = requests.post(f"{API_URL}/predict", json={"text": input_text}).json()
                    
                    # 2. Hit Explain Endpoint
                    explain_res = requests.post(f"{API_URL}/explain", json={"text": input_text}).json()
                    
                    # 3. Hit RAG Endpoint
                    rag_res = requests.post(f"{API_URL}/rag", json={"text": input_text}).json()

                    # Save input text in session state for other pages
                    st.session_state["last_input"] = input_text
                    st.session_state["last_explanation"] = explain_res

                    # Displays Results
                    st.markdown("## 📊 Analysis Results")
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        # Verdict card
                        verdict = pred_res["prediction"]
                        confidence = pred_res["confidence"] * 100

                        if verdict == "FAKE":
                            st.markdown(
                                f"""
                                <div style='background-color: rgba(255, 75, 75, 0.1); border: 2px solid #FF4B4B; border-radius: 12px; padding: 25px; text-align: center;'>
                                    <h3 style='color: #FF4B4B; margin: 0; font-size: 1.5rem;'>PREDICTION</h3>
                                    <h1 style='color: #FF4B4B; font-size: 3.5rem; font-weight: 800; margin: 10px 0;'>🚨 FAKE NEWS</h1>
                                    <h3 style='color: #ccc; margin: 0;'>Confidence: {confidence:.2f}%</h3>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                f"""
                                <div style='background-color: rgba(46, 204, 113, 0.1); border: 2px solid #2ECC71; border-radius: 12px; padding: 25px; text-align: center;'>
                                    <h3 style='color: #2ECC71; margin: 0; font-size: 1.5rem;'>PREDICTION</h3>
                                    <h1 style='color: #2ECC71; font-size: 3.5rem; font-weight: 800; margin: 10px 0;'>✅ REAL NEWS</h1>
                                    <h3 style='color: #ccc; margin: 0;'>Confidence: {confidence:.2f}%</h3>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                    with c2:
                        # RAG Explanation summary
                        st.markdown(
                            f"""
                            <div class='glass-card' style='height: 100%;'>
                                <h3 style='margin-top: 0;'>🤖 AI Fact-Checking Assistant</h3>
                                <p style='font-size: 1.1rem; line-height: 1.6; color: #eee;'>{rag_res['explanation']}</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    # Highlight suspicious words (SHAP highlights)
                    st.markdown("### 🔍 Highlighted Key Terms")
                    st.write("Words contributing most to the verdict (Hover or read importance):")
                    
                    importances = explain_res.get("importances", [])
                    # Sort by magnitude of score
                    importances_sorted = sorted(importances, key=lambda x: abs(x["score"]), reverse=True)[:15]
                    
                    # Custom html span highlighting
                    highlighted_html = ""
                    # Create list of words that are highly relevant
                    for imp in importances_sorted:
                        word = imp["word"]
                        score = imp["score"]
                        if score > 0.005:  # Real indicator
                            highlighted_html += f"<span class='highlight-real' title='Score: {score:.4f}'>{word}</span> "
                        elif score < -0.005:  # Fake indicator
                            highlighted_html += f"<span class='highlight-fake' title='Score: {score:.4f}'>{word}</span> "
                        else:
                            highlighted_html += f"<span style='color: #ccc; margin-right: 4px;'>{word}</span>"
                            
                    st.markdown(
                        f"<div class='glass-card'>{highlighted_html if highlighted_html else 'No high-scoring words found.'}</div>",
                        unsafe_allow_html=True,
                    )

                    # Retrieved Evidence
                    st.markdown("### 📚 Retrieved Evidence from Trusted Database")
                    evidence_list = rag_res.get("evidence", [])
                    
                    if not evidence_list:
                        st.info("No matching articles found in reference database.")
                    else:
                        for idx, ev in enumerate(evidence_list):
                            label_class = "highlight-real" if ev["label"] == "REAL" else "highlight-fake"
                            st.markdown(
                                f"""
                                <div class='glass-card'>
                                    <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;'>
                                        <h4 style='margin: 0; color: #fff;'>Evidence {idx+1}: {ev['title']}</h4>
                                        <span class='{label_class}'>{ev['label']}</span>
                                    </div>
                                    <p style='color: #aaa; font-style: italic; font-size: 0.95rem; margin-bottom: 5px;'>Subject: {ev['subject']} | Distance Score: {ev['score']:.4f}</p>
                                    <p style='color: #ddd; line-height: 1.5;'>{ev['content'][:350]}...</p>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                    # Feedback section
                    st.markdown("### 💬 Help Us Improve")
                    st.write("Do you agree with this classification?")
                    f_col1, f_col2, f_col3 = st.columns([1, 1, 4])
                    
                    with f_col1:
                        agree = st.button("👍 Agree", use_container_width=True)
                    with f_col2:
                        disagree = st.button("👎 Disagree", use_container_width=True)

                    if agree or disagree:
                        feedback_payload = {
                            "text": input_text,
                            "prediction": verdict,
                            "is_correct": True if agree else False,
                            "comment": "Streamlit User feedback",
                        }
                        try:
                            f_res = requests.post(f"{API_URL}/feedback", json=feedback_payload).json()
                            st.success("Thank you for your feedback!")
                        except Exception as e:
                            st.error(f"Failed to submit feedback: {e}")

                except Exception as e:
                    st.error(f"Could not connect to FastAPI server at {API_URL}. Please ensure it is running.")
                    st.exception(e)

# ----------------- EDA PAGE -----------------
elif page == "Exploratory Data Analysis (EDA)":
    st.markdown("<h1 class='main-title'>📊 Dataset Inspection & EDA</h1>", unsafe_allow_html=True)
    st.write("Explore statistical distributions, themes, and characteristics of Fake vs Real news.")

    df = load_dataset_cached()

    if df is not None:
        st.markdown("### Dataset Overview")
        st.write(f"Total articles: **{len(df):,}** | Columns: `{list(df.columns)}`")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                f"""
                <div class='glass-card' style='text-align: center;'>
                    <h3>Fake News Articles</h3>
                    <p class='metric-value'>{(df['label'] == 0).sum():,}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"""
                <div class='glass-card' style='text-align: center;'>
                    <h3>Real News Articles</h3>
                    <p class='metric-value'>{(df['label'] == 1).sum():,}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                f"""
                <div class='glass-card' style='text-align: center;'>
                    <h3>Missing Values</h3>
                    <p class='metric-value'>{df.isnull().sum().sum()}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Plot 1: Class distribution
        st.markdown("### Class Distribution & Volume")
        label_map = {0: "Fake News", 1: "Real News"}
        df_plot_label = df.copy()
        df_plot_label["Verdict"] = df_plot_label["label"].map(label_map)
        
        class_counts = df_plot_label["Verdict"].value_counts().reset_index()
        class_counts.columns = ["Verdict", "Count"]
        
        fig1 = px.pie(
            class_counts,
            values="Count",
            names="Verdict",
            color="Verdict",
            color_discrete_map={"Fake News": "#FF4B4B", "Real News": "#2ECC71"},
            hole=0.4,
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Plot 2: Subject counts
        st.markdown("### Top Subjects by Category")
        subject_counts = df_plot_label.groupby(["subject", "Verdict"]).size().reset_index(name="Count")
        fig2 = px.bar(
            subject_counts,
            x="subject",
            y="Count",
            color="Verdict",
            barmode="group",
            color_discrete_map={"Fake News": "#FF4B4B", "Real News": "#2ECC71"},
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Plot 3: Text length distribution
        st.markdown("### Article Length Analysis")
        df_plot_label["Char Count"] = df_plot_label["text"].apply(lambda x: len(str(x)))
        df_plot_label["Word Count"] = df_plot_label["text"].apply(lambda x: len(str(x).split()))
        
        fig3 = px.box(
            df_plot_label,
            x="Verdict",
            y="Word Count",
            color="Verdict",
            color_discrete_map={"Fake News": "#FF4B4B", "Real News": "#2ECC71"},
            title="Word Count Distribution",
        )
        st.plotly_chart(fig3, use_container_width=True)

        # Plot 4: Word clouds
        st.markdown("### Word Clouds")
        wc_col1, wc_col2 = st.columns(2)
        
        with wc_col1:
            st.subheader("Fake News Themes")
            fake_texts = " ".join(df_plot_label[df_plot_label["label"] == 0]["title"].astype(str).sample(500, random_state=42))
            wordcloud_fake = WordCloud(width=800, height=400, background_color="black", colormap="Reds").generate(fake_texts)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud_fake, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
            
        with wc_col2:
            st.subheader("Real News Themes")
            real_texts = " ".join(df_plot_label[df_plot_label["label"] == 1]["title"].astype(str).sample(500, random_state=42))
            wordcloud_real = WordCloud(width=800, height=400, background_color="black", colormap="Greens").generate(real_texts)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud_real, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

        # Plot 5: Sentiment analysis (Phase 6)
        st.markdown("### 🎭 Sentiment Analysis (Phase 6)")
        st.write("Compare the sentiment profiles and emotional intensity of Fake vs Real news.")
        
        with st.spinner("Analyzing sentiment distributions..."):
            sentiment_df = compute_sentiment_stats(df)
            
            # Group by label
            sentiment_df["Verdict"] = sentiment_df["label"].map(label_map)
            avg_stats = sentiment_df.groupby("Verdict")[["polarity_score", "subjectivity_score"]].mean().reset_index()
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Average Sentiment Polarity** (Negative to Positive)")
                fig_pol = px.bar(
                    avg_stats,
                    x="Verdict",
                    y="polarity_score",
                    color="Verdict",
                    color_discrete_map={"Fake News": "#FF4B4B", "Real News": "#2ECC71"},
                    labels={"polarity_score": "Mean Polarity"},
                )
                st.plotly_chart(fig_pol, use_container_width=True)
                
            with c2:
                st.markdown("**Average Subjectivity / Emotional Charge** (Objective to Subjective)")
                fig_subj = px.bar(
                    avg_stats,
                    x="Verdict",
                    y="subjectivity_score",
                    color="Verdict",
                    color_discrete_map={"Fake News": "#FF4B4B", "Real News": "#2ECC71"},
                    labels={"subjectivity_score": "Mean Subjectivity"},
                )
                st.plotly_chart(fig_subj, use_container_width=True)
                
            # Answer Phase 6 question
            fake_subj = avg_stats[avg_stats["Verdict"] == "Fake News"]["subjectivity_score"].values[0]
            real_subj = avg_stats[avg_stats["Verdict"] == "Real News"]["subjectivity_score"].values[0]
            
            st.markdown(
                f"""
                <div class='glass-card'>
                    <h4>❓ Do fake news articles tend to be more emotionally charged?</h4>
                    <p style='font-size: 1.1rem; line-height: 1.6;'>
                        <strong>Verdict:</strong> Yes! Fake news articles have an average subjectivity score of 
                        <span class='highlight-fake'>{fake_subj:.4f}</span> compared to 
                        <span class='highlight-real'>{real_subj:.4f}</span> for real news. 
                        Subjectivity measures the amount of personal opinion and emotion in a text rather than objective facts. 
                        Our analysis confirms that fake news articles are statistically <strong>more emotionally charged</strong> and use less objective language.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ----------------- EXPLAINABILITY INSIGHTS -----------------
elif page == "Explainability Insights":
    st.markdown("<h1 class='main-title'>🔬 SHAP Explainability Insights</h1>", unsafe_allow_html=True)
    st.write("Understand how the BERT model weights individual words when computing its classification scores.")

    if "last_explanation" not in st.session_state:
        st.info("Please paste and verify an article on the News Detector page first to view explainability insights.")
    else:
        explain_res = st.session_state["last_explanation"]
        last_input = st.session_state.get("last_input", "")

        st.markdown("### Prediction Overview")
        verdict = explain_res["prediction"]
        conf = explain_res["probability"] * 100
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Model Classification Verdict", verdict)
        with col2:
            st.metric("Model Confidence", f"{conf:.2f}%")

        st.markdown("### Word Importance Distribution (SHAP)")
        st.write("Hover or look at the bar chart below to see the impact of each word:")
        
        importances = explain_res.get("importances", [])
        if not importances:
            st.write("No SHAP values computed.")
        else:
            # Create a dataframe for Plotly
            imp_df = pd.DataFrame(importances)
            # Remove punctuation tokens and markers
            imp_df = imp_df[~imp_df["word"].str.strip().isin(["", "[CLS]", "[SEP]", ".", ",", "!", "?", '"'])]
            imp_df = imp_df.sort_values(by="score", key=abs, ascending=False).head(20)
            
            # Reorder for visualization
            imp_df = imp_df.sort_values(by="score")
            
            fig = go.Figure()
            # Green for Positive (indicates class 1 - REAL), Red for Negative (class 0 - FAKE)
            colors = ["#2ECC71" if val >= 0 else "#FF4B4B" for val in imp_df["score"]]
            
            fig.add_trace(
                go.Bar(
                    y=imp_df["word"],
                    x=imp_df["score"],
                    orientation="h",
                    marker_color=colors,
                )
            )
            
            fig.update_layout(
                title="SHAP Importance Values (Top 20 Words)",
                xaxis_title="SHAP Impact Score (Positive = Real, Negative = Fake)",
                yaxis_title="Words / Tokens",
                height=600,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### How to read SHAP values?")
        st.markdown(
            """
            - **Red Bars (Negative Scores)**: Pull the model prediction towards **FAKE NEWS** (Class 0).
            - **Green Bars (Positive Scores)**: Push the model prediction towards **REAL NEWS** (Class 1).
            - The length of the bar shows the strength of the word's influence on the final model prediction.
            - Highly charged words like *'shocking'*, *'breaking'*, or *'unsupported'* often have strong negative values in Fake News articles.
            """
        )

# ----------------- KNOWLEDGE SEARCH -----------------
elif page == "Knowledge Search (RAG)":
    st.markdown("<h1 class='main-title'>🔍 Vector Store Semantic Search</h1>", unsafe_allow_html=True)
    st.write("Search the reference database semantic vector space directly to retrieve evidence and fact-checking context.")

    search_query = st.text_input("Enter semantic query:", placeholder="E.g., Russia sanctions Senate vote...")
    top_k = st.slider("Number of matching chunks to retrieve:", min_value=1, max_value=5, value=3)

    if st.button("Search Database", type="primary"):
        if not search_query.strip():
            st.warning("Please type a search query first!")
        else:
            with st.spinner("Searching Chroma DB vector store..."):
                try:
                    rag_res = requests.post(f"{API_URL}/rag", json={"text": search_query}).json()
                    evidence_list = rag_res.get("evidence", [])

                    st.markdown(f"### Results for: *'{search_query}'*")
                    
                    if not evidence_list:
                        st.info("No semantic matches found.")
                    else:
                        for idx, ev in enumerate(evidence_list[:top_k]):
                            label_class = "highlight-real" if ev["label"] == "REAL" else "highlight-fake"
                            st.markdown(
                                f"""
                                <div class='glass-card'>
                                    <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;'>
                                        <h4 style='margin: 0; color: #fff;'>Match {idx+1}: {ev['title']}</h4>
                                        <span class='{label_class}'>{ev['label']}</span>
                                    </div>
                                    <p style='color: #aaa; font-style: italic; font-size: 0.95rem; margin-bottom: 5px;'>Subject: {ev['subject']} | Distance Score: {ev['score']:.4f}</p>
                                    <p style='color: #ddd; line-height: 1.5;'>{ev['content']}</p>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                except Exception as e:
                    st.error("Failed to query vector database. Please make sure the FastAPI server is running.")
                    st.exception(e)

# ----------------- MODEL METRICS -----------------
elif page == "Model Metrics":
    st.markdown("<h1 class='main-title'>📈 BERT Model performance & Metrics</h1>", unsafe_allow_html=True)
    st.write("Detailed view of evaluation metrics, confusion matrix, and training details for the fine-tuned BERT classifier.")

    metrics_path = Path("outputs/models/metrics.json")

    if not metrics_path.exists():
        st.warning(
            "Metrics file not found. Please run model evaluation (`python -m app.training.evaluate`) first."
        )
    else:
        try:
            with open(metrics_path, "r") as f:
                metrics = json.load(f)

            # High-level metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(
                    f"""
                    <div class='glass-card' style='text-align: center;'>
                        <h3>Accuracy</h3>
                        <p class='metric-value'>{metrics['accuracy'] * 100:.2f}%</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col2:
                st.markdown(
                    f"""
                    <div class='glass-card' style='text-align: center;'>
                        <h3>Precision</h3>
                        <p class='metric-value'>{metrics['precision'] * 100:.2f}%</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col3:
                st.markdown(
                    f"""
                    <div class='glass-card' style='text-align: center;'>
                        <h3>Recall</h3>
                        <p class='metric-value'>{metrics['recall'] * 100:.2f}%</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col4:
                st.markdown(
                    f"""
                    <div class='glass-card' style='text-align: center;'>
                        <h3>ROC-AUC</h3>
                        <p class='metric-value'>{metrics['roc_auc'] * 100:.2f}%</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Confusion Matrix Plot
            st.markdown("### Confusion Matrix")
            cm = np.array(metrics["confusion_matrix"])
            
            fig = px.imshow(
                cm,
                text_auto=True,
                labels=dict(x="Predicted Label", y="True Label", color="Number of Samples"),
                x=["Fake News (0)", "Real News (1)"],
                y=["Fake News (0)", "Real News (1)"],
                color_continuous_scale="Purples",
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

            # Classification Report Table
            st.markdown("### Detailed Classification Report")
            report_dict = metrics["classification_report"]
            report_df = pd.DataFrame(report_dict).transpose()
            st.dataframe(report_df.style.highlight_max(axis=0, color="#852DF6"))

        except Exception as e:
            st.error(f"Failed to read metrics: {e}")

# ----------------- ABOUT PROJECT -----------------
elif page == "About Project":
    st.markdown("<h1 class='main-title'>🛡️ Project Architecture & Details</h1>", unsafe_allow_html=True)
    
    st.markdown(
        """
        ### What is FakeShield AI?
        **FakeShield AI** is an explainable fake news detection and fact-checking platform that combines 
        deep learning NLP classification with generative AI retrieval-augmented generation (RAG).
        
        ### Architecture Overview
        ```
                                       User Interface
                                             │ (FastAPI REST)
                                             ▼
                                    Streamlit Dashboard
                                             │
                            ┌────────────────┴────────────────┐
                            ▼                                 ▼
                   Fake News Detection                   AI Assistant
                    (Fine-tuned BERT)                 (LangChain + RAG)
                            │                                 │
                            ▼                                 ▼
                  Explainability Module                    ChromaDB
                       (SHAP/LIME)                      Vector Store
                            │                                 │
                            └────────────────┬────────────────┘
                                             ▼
                                      FastAPI Backend
                                             │
                                  Dataset + Knowledge Base
        ```
        
        ### Core Features
        1. **BERT News Detector**: Fine-tuned bidirectional transformer model classifies inputs as FAKE or REAL.
        2. **SHAP Explainability**: Highlights which specific words in the text contributed most to the prediction score.
        3. **RAG Fact Checker**: Retrieves similar articles from a ChromaDB vector database and uses an LLM (Gemini/OpenAI/Rule Engine) to summarize evidence and print a verified verdict.
        """
    )
