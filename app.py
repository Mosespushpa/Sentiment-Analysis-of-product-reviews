import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# --- Initialization ---
@st.cache_resource
def setup_nlp():
    try:
        nltk.download('stopwords')
    except:
        pass
    return SentimentIntensityAnalyzer(), set(stopwords.words('english'))

analyzer, stop_words = setup_nlp()

def preprocess_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'[^\w\s]', '', text).lower()
    return ' '.join([word for word in text.split() if word not in stop_words])

def get_sentiment(rev):
    score = analyzer.polarity_scores(rev)['compound']
    if score >= 0.05: return 1  # Positive
    elif score <= -0.05: return 0  # Negative
    return 2  # Neutral

# --- Page Setup ---
st.set_page_config(page_title="iPhone Sentiment Tool", layout="wide")

# CSS for Pink/Black theme
st.markdown("""
    <style>
    .stApp { background: linear-gradient(to bottom, #FFC0CB, #000000); }
    h1, h2, h3, p, span, label, .stMarkdown { color: white !important; }
    .stMetric { background: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 10px; border: 1px solid pink; }
    .sidebar .sidebar-content { background: rgba(0,0,0,0.5); }
    </style>
    """, unsafe_allow_html=True)

# --- Sidebar: System Analysis & File Upload ---
st.sidebar.title("Project System Analysis")
st.sidebar.info("""
**Objective:** Feature-Specific Sentiment Analysis  
**Model:** SVM Classifier + VADER  
**Features:** Camera, Battery, Performance, Screen, Design, Price
""")

st.sidebar.header("📂 Data Source")
uploaded_file = st.sidebar.file_uploader("Upload your own CSV", type=["csv"])

if uploaded_file is not None:
    data_path = uploaded_file
    st.sidebar.success("Using Uploaded Dataset")
else:
    data_path = 'iphone15-reviews.csv'
    st.sidebar.info("Using Default: apple-iphone-16-128-gb_reviews.csv")

# --- Load & Process Data ---
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    # Ensure 'review' column exists
    if 'review' not in df.columns:
        st.error("CSV must have a 'review' column!")
        st.stop()
    df['cleaned_review'] = df['review'].apply(preprocess_text)
    df['sentiment'] = df['cleaned_review'].apply(get_sentiment)
    return df.dropna(subset=['cleaned_review', 'sentiment'])

try:
    df = load_data(data_path)
except FileNotFoundError:
    st.error("Default file not found. Please upload a CSV.")
    st.stop()

# --- ML Model Training ---
tfidf = TfidfVectorizer(max_features=1000)
X = tfidf.fit_transform(df['cleaned_review']).toarray()
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# --- Main UI ---
st.title("📱 iPhone Feature Sentiment Analysis")

tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "☁️ Word Clouds", "🔍 Predict"])

with tab1:
    st.header("Feature-Specific Sentiment")
    aspects = ['battery', 'camera', 'performance', 'screen', 'design', 'price']
    
    aspect_results = []
    for a in aspects:
        mask = df['cleaned_review'].str.contains(a)
        p = len(df[mask & (df['sentiment'] == 1)])
        n = len(df[mask & (df['sentiment'] == 0)])
        aspect_results.append({'Aspect': a, 'Positive': p, 'Negative': n})
    
    res_df = pd.DataFrame(aspect_results)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('none')
    ax.bar(res_df['Aspect'], res_df['Positive'], label='Positive', color='#2ecc71')
    ax.bar(res_df['Aspect'], res_df['Negative'], bottom=res_df['Positive'], label='Negative', color='#e74c3c')
    ax.set_facecolor('none')
    ax.tick_params(colors='white')
    plt.legend()
    st.pyplot(fig)

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Positive Reviews")
        pos_words = ' '.join(df[df['sentiment'] == 1]['cleaned_review'])
        if pos_words:
            wc = WordCloud(width=400, height=300, background_color='white').generate(pos_words)
            st.image(wc.to_array())
    with col2:
        st.subheader("Negative Reviews")
        neg_words = ' '.join(df[df['sentiment'] == 0]['cleaned_review'])
        if neg_words:
            wc = WordCloud(width=400, height=300, background_color='black', colormap='Reds').generate(neg_words)
            st.image(wc.to_array())

with tab3:
    st.header("Test a Review")
    user_text = st.text_input("Enter text (e.g., 'The camera is amazing but battery sucks'):")
    if user_text:
        processed = preprocess_text(user_text)
        vec = tfidf.transform([processed]).toarray()
        pred = svm.predict(vec)[0]
        label = "Positive ✅" if pred == 1 else ("Negative ❌" if pred == 0 else "Neutral 😐")
        st.subheader(f"Result: {label}")
