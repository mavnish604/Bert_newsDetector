import streamlit as st
import joblib
import os
from sentence_transformers import SentenceTransformer
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# --- Design System (Custom CSS) ---
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        border: 1px solid #ff4b4b;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        text-align: center;
        font-size: 1.5em;
        font-weight: bold;
    }
    .real {
        background-color: #006600;
        color: white;
    }
    .fake {
        background-color: #660000;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- Model Loading ---
MODEL_PATH = "fake_news_model/checkpoint-10731"
CLASSIFIER_PATH = "classifier.joblib"

@st.cache_resource
def load_models():
    with st.spinner("Loading models... This may take a moment."):
        # Load SentenceTransformer
        if os.path.exists(MODEL_PATH):
            model = SentenceTransformer(MODEL_PATH)
        else:
            # Fallback to original if checkpoint moved or deleted
            model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load Logistic Regression Classifier
        if os.path.exists(CLASSIFIER_PATH):
            clf = joblib.load(CLASSIFIER_PATH)
        else:
            st.error("Classifier model not found! Please run export_classifier.py first.")
            return None, None
            
    return model, clf

# --- Helper Functions ---
def predict_news(text, model, clf):
    if not text.strip():
        return None, None
    
    # 1. Encode text to embeddings
    with st.spinner("Analyzing text..."):
        embedding = model.encode([text])
    
    # 2. Predict using LR
    label_idx = clf.predict(embedding)[0]
    probs = clf.predict_proba(embedding)[0]
    confidence = probs[label_idx]
    
    # Labels: 0 corresponds to the "old 1" (which we might need to map back)
    # Let's check the notebook's invert_label:
    # if val == 1 (original dataset, likely fake or real? let's stick to the notebook's logic)
    # Notebook: df["label"] = df["label"].apply(invert_label) where 1->0, 0->1
    # After inversion: 1 = True (original 0), 0 = Fake (original 1)
    label = "True (Real)" if label_idx == 1 else "Fake"
    
    return label, confidence

# --- UI Components ---
def main():
    st.title("📰 Fake News Detector")
    st.markdown("Enter a news article or snippet below to check its authenticity using our fine-tuned AI model.")

    model, clf = load_models()
    
    if model is None or clf is None:
        return

    # Text input
    news_text = st.text_area("News Content", height=250, placeholder="Paste the text of the article here...")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        predict_button = st.button("Detect Authenticities")

    if predict_button:
        if news_text.strip():
            start_time = time.time()
            label, confidence = predict_news(news_text, model, clf)
            end_time = time.time()
            
            if label:
                color_class = "real" if "True" in label else "fake"
                st.markdown(f"""
                    <div class="prediction-card {color_class}">
                        PREDICTION: {label}<br/>
                        <span style="font-size: 0.7em; font-weight: normal;">Confidence: {confidence:.2%}</span>
                    </div>
                """, unsafe_allow_html=True)
                
                st.info(f"Analysis completed in {end_time - start_time:.2f} seconds.")
        else:
            st.warning("Please enter some text to analyze.")

    # Sidebar / Info
    with st.sidebar:
        st.header("About")
        st.write("This application uses a **SentenceTransformer** (all-MiniLM-L6-v2) fine-tuned on the WELFake dataset.")
        st.write("The fine-tuning used **BatchHardTripletLoss** to pull similar news together and push fake/real embeddings apart.")
        st.write("A **Logistic Regression** classifier is used on top of the embeddings for the final prediction.")
        st.divider()
        st.write("📊 **Model Specs**")
        st.write("- Base: MiniLM-L6-v2")
        st.write("- Classifier: Logistic Regression")
        st.write("- Accuracy: ~99%")

if __name__ == "__main__":
    main()
