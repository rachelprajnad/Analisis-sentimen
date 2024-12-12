import streamlit as st
import pandas as pd
import spacy
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time
import seaborn as sns
from google_play_scraper import reviews, Sort
import matplotlib.pyplot as plt
from wordcloud import WordCloud




# Pastikan resource NLTK dan SpaCy sudah terinstal
nltk.download('stopwords')
nltk.download('punkt')

# Muat model SpaCy
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
analyser = SentimentIntensityAnalyzer()

# Fungsi untuk preprocessing data
def case_folding(text):
    return text.lower()

def cleaning(text):
    text = re.sub(r"[^a-zA-Z0-9\s_apos_]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r"[\t\n\r\\]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def fix_contractions(text):
    contractions_dict = {
        r"\bI'm\b": "I am", r"\bI've\b": "I have", r"\bIt's\b": "It is",
        r"\bYou're\b": "You are", r"\bHe's\b": "He is", r"\bShe's\b": "She is",
        r"\bWe've\b": "We have", r"\bThey're\b": "They are", r"\bCan't\b": "Cannot",
        r"\bDon't\b": "Do not", r"\bDidn't\b": "Did not", r"\bIsn't\b": "Is not",
        r"\bWasn't\b": "Was not", r"\bAren't\b": "Are not"
    }
    for contraction, fixed in contractions_dict.items():
        text = re.sub(contraction, fixed, text)
    return text

def process_tokenization(text):
    text = fix_contractions(text)  # Perbaiki kontraksi terlebih dahulu
    tokens = re.findall(r"\b\w+'\w+|\w+|[^\w\s]", text)  # Tokenisasi menggunakan regex
    return tokens

def normalize(tokens):
    normalization_dict = {
        'thanks': 'thank you', 'pls': 'please', 'thx': 'thank you', 'sorry': 'apologies'
    }
    return [normalization_dict.get(token, token) for token in tokens]

def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words and word not in string.punctuation]

def lemmatize(tokens):
    return [nlp(word)[0].lemma_ for word in tokens if word not in stop_words and word not in string.punctuation]

def preprocess_text(text):
    text = case_folding(text)
    text = cleaning(text)
    tokens = process_tokenization(text)
    tokens = normalize(tokens)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    return ' '.join(tokens)

def get_sentiment_and_score(text):
    score = analyser.polarity_scores(text)['compound']
    if score >= 0.05:
        sentiment = 'positive'
    elif score <= -0.05:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    
    return score, sentiment

# Streamlit UI untuk scraping
def scraping():
    st.title("🔍 Scraping Data")
    st.write("Pilih aplikasi dan jumlah data untuk melakukan scraping.")
    
    apps = {
        "Instagram": "com.instagram.android",
        "TikTok": "com.zhiliaoapp.musically",
        "WhatsApp": "com.whatsapp",
        "Facebook": "com.facebook.katana",
        "Telegram": "org.telegram.messenger"
    }
    
    apk_name = st.selectbox("Pilih Aplikasi", list(apps.keys()))
    num_data = st.number_input("Jumlah Data", min_value=10, max_value=5000, step=10, value=100)

    if st.button("Submit"):
        st.write(f"Scraping ulasan {apk_name}...")

        package_name = apps[apk_name]
        total_reviews = []
        batch_size = 100
        for i in range(0, num_data, batch_size):
            try:
                reviews_data, _ = reviews(package_name, lang='en', country='us', sort=Sort.MOST_RELEVANT, count=batch_size)
                total_reviews.extend(reviews_data)
                if len(total_reviews) >= num_data:
                    break
            except Exception as e:
                st.error(f"Terjadi kesalahan saat scraping: {e}")
                break

        if not total_reviews:
            st.write("Tidak ada ulasan yang ditemukan.")
            return

        df = pd.DataFrame(total_reviews)
        st.write("Data Ulasan yang Diperoleh (Filtered):")
        st.dataframe(df[['reviewId', 'score', 'content', 'appVersion']])

        csv_file = "filtered_reviews.csv"
        df[['reviewId', 'score', 'content', 'appVersion']].to_csv(csv_file, index=False)
        with open(csv_file, "rb") as f:
            st.download_button("📥Download Filtered CSV", data=f, file_name=csv_file)

# Menambahkan Visualisasi Sentimen
def plot_sentiment(df):
    sentiment_scores = df['score'].apply(lambda x: get_sentiment_and_score(x)[0])
    sentiment_labels = ['Positive' if score >= 0.05 else 'Negative' if score <= -0.05 else 'Neutral' for score in sentiment_scores]
    sentiment_df = pd.DataFrame({'Sentiment': sentiment_labels})

    # Plot Sentiment Distribution
    sns.countplot(x='Sentiment', data=sentiment_df)
    plt.title('Distribusi Sentimen Ulasan')
    st.pyplot()

# Halaman Preprocessing di Streamlit
def preprocessing():
    st.title("⚙️ Preprocessing & Labeling")
    st.write("Unggah file CSV untuk melakukan preprocessing data.")
    
    uploaded_file = st.file_uploader("Upload File CSV", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Data yang diunggah:")
        st.write(data.head())

        if st.button("🛠️ Start Preprocessing"):
            st.write("Memulai preprocessing...")

            # Progress bar
            progress_bar = st.progress(0)

            for i in range(len(data)):
                progress_bar.progress((i+1) / len(data))
                data['cleaned'] = data['content'].apply(preprocess_text)

            st.write("Proses Preprocessing selesai!")
            st.dataframe(data.head())

# Halaman utama
def home():
    st.title("Analisis Sentimen Instagram")
    st.markdown("""
        🌟 **Selamat datang di Aplikasi Analisis Sentimen Instagram!** 🌟
        Aplikasi ini dirancang untuk membantu Anda melakukan analisis sentimen ulasan aplikasi Instagram yang diambil dari **Google Play Store**.
        ...
    """)
