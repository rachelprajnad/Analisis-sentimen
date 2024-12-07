import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from google_play_scraper import Sort, reviews
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import spacy
import time
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from wordcloud import WordCloud
import spacy
from spacy.cli import download

# Download dan instal model
download("en_core_web_sm")

# Pastikan resource NLTK dan SpaCy sudah terinstal
nltk.download('stopwords')
nltk.download('punkt')

# Inisialisasi model SpaCy dan utilitas lainnya
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
analyser = SentimentIntensityAnalyzer()

# Fungsi Case Folding
def case_folding(text):
    return text.lower()

# Fungsi Cleaning
def cleaning(text):
    text = re.sub(r"[^a-zA-Z0-9\s_apos_]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r"[\t\n\r\\]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Fungsi untuk memperbaiki kontraksi
def fix_contractions(text):
    contractions_dict = {
        r"\bI'm\b": "I am", r"\bI've\b": "I have", r"\bIt's\b": "It is",
        r"\bYou're\b": "You are", r"\bHe's\b": "He is", r"\bShe's\b": "She is",
        r"\bWe've\b": "We have", r"\bThey're\b": "They are", r"\bCan't\b": "Cannot",
        r"\bDon't\b": "Do not", r"\bDidn't\b": "Did not", r"\bIsn't\b": "Is not",
        r"\bWasn't\b": "Was not", r"\bAren't\b": "Are not",
        # Menambahkan aturan kontraksi tanpa apostrof
        r"\bI m\b": "I am",  # Mengganti "I m" dengan "I am"
        r"\bIt s\b": "It is",  # Mengganti "it s" dengan "It is"
        r"\bYou re\b": "You are",  # Mengganti "you re" dengan "you are"
        r"\bHe s\b": "He is",  # Mengganti "he s" dengan "he is"
        r"\bShe s\b": "She is",  # Mengganti "she s" dengan "she is"
    }
    
    # Terapkan regex untuk mengganti kontraksi dengan kata yang benar
    for contraction, fixed in contractions_dict.items():
        text = re.sub(contraction, fixed, text)

    return text

# Tokenisasi dengan perbaikan kontraksi menggunakan regex
def process_tokenization(text):
    text = fix_contractions(text)  # Perbaiki kontraksi terlebih dahulu
    tokens = re.findall(r"\b\w+'\w+|\w+|[^\w\s]", text)  # Tokenisasi menggunakan regex
    return tokens

# Fungsi Normalisasi (penggantian slang atau kata informal)
normalization_dict = {
    'thanks': 'thank you', 'pls': 'please', 'thx': 'thank you','sorry': 'apologies', 'omg': 'oh my god','lol': 'laugh out loud',
    'brb': 'be right back', 'btw': 'by the way', 'idk': 'i don’t know', 'tbh': 'to be honest', 'lmao': 'laughing my ass off',
    'fyi': 'for your information', 'sry': 'sorry', 'smh': 'shaking my head', 'rofl': 'rolling on the floor laughing',
    'asap': 'as soon as possible', 'bff': 'best friends forever', 'fomo': 'fear of missing out', 'yolo': 'you only live once',
    'gtg': 'got to go', 'nvm': 'never mind', 'np': 'no problem', 'cya': 'see you', 'gg': 'good game', 'wfh': 'work from home',
    'ftw': 'for the win','lmk': 'let me know','tmi': 'too much information','bbl': 'be back later', 'mfw': 'my face when',
    'imo': 'in my opinion','atm': 'at the moment',}

def normalize(tokens):
    return [normalization_dict.get(token, token) for token in tokens]

# Stopwords Removal
def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words and word not in string.punctuation]

# Fungsi Lemmatization dengan SpaCy
def lemmatize(tokens):
    # Menggunakan lemmatization dari SpaCy
    return [nlp(word)[0].lemma_ for word in tokens if word not in stop_words and word not in string.punctuation]

# Fungsi utama Preprocessing (Otomatis Semua Langkah)
def preprocess_text(text):
    text = case_folding(text)
    text = cleaning(text)
    tokens = process_tokenization(text)  # Menggunakan process_tokenization yang baru
    tokens = normalize(tokens)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    return ' '.join(tokens)  # Rekonstruksi kalimat dari token

# Fungsi Labeling dengan VADER
def get_sentiment_and_score(text):
    score = analyser.polarity_scores(text)['compound']
    if score >= 0.05:
        sentiment = 'positive'
    elif score <= -0.05:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    
    return score, sentiment

def home():
    # Judul dengan emoji yang menarik
    st.title("Analisis Sentimen Instagram")
    
    # Deskripsi lebih menarik dengan beberapa elemen gaya
    st.markdown(
        """
        🌟 **Selamat datang di Aplikasi Analisis Sentimen Instagram!** 🌟


        Aplikasi ini dirancang untuk membantu Anda melakukan analisis sentimen ulasan aplikasi Instagram yang diambil dari **Google Play Store**. Dengan menggunakan metode **VADER** (Valence Aware Dictionary and sentiment Reasoner), kami memberikan wawasan mengenai perasaan pengguna terhadap aplikasi Instagram!

        🚀 **Fitur utama aplikasi:**
        - **Scraping Data**: Ambil ulasan terbaru dari pengguna Instagram di Google Play Store.
        - **Preprocessing Data**: Bersihkan dan persiapkan data untuk analisis.
        - **Visualisasi Sentimen**: Lihat distribusi sentimen positif, netral, dan negatif dari ulasan.
        - **Prediksi Sentimen**: Masukkan ulasan Anda untuk mengetahui sentimen yang terkandung di dalamnya.
        - **Analisis Tren Waktu**: Lihat bagaimana sentimen pengguna berubah berdasarkan versi aplikasi dan waktu.

        💡 Dengan aplikasi ini, Anda dapat memahami bagaimana perasaan pengguna terhadap aplikasi Instagram pada berbagai aspek.

        🛠️ **Cara Penggunaan**:
        1. **Scraping Data**: Ambil ulasan aplikasi Instagram.
        2. **Preprocessing**: Bersihkan dan siapkan data.
        3. **Analisis Sentimen**: Lihat bagaimana pengguna merasa tentang aplikasi Instagram.
        
        **Ayo Mulai!** Klik menu di samping untuk memulai perjalanan analisis Anda! 🔍
        """
    )

    # Menambahkan gambar visual atau ilustrasi terkait Instagram untuk tampilan yang lebih menarik
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/9/95/Instagram_logo_2022.svg/1024px-Instagram_logo_2022.svg.png", width=200)

    # Link untuk mengarahkan pengguna ke halaman lain atau referensi lebih lanjut
    st.markdown(
        """
        👉 **Ingin tahu lebih banyak tentang analisis sentimen dan metode VADER?** 
        [Baca lebih lanjut di sini!](https://en.wikipedia.org/wiki/VADER)
        """
    )

    

  



def scraping():
    st.title("🔍 Scraping Data")
    st.write("Pilih aplikasi dan jumlah data untuk melakukan scraping.")
    
    # Daftar aplikasi dengan nama paket
    apps = {
        "Instagram": "com.instagram.android",
        "TikTok": "com.zhiliaoapp.musically",
        "WhatsApp": "com.whatsapp",
        "Facebook": "com.facebook.katana",
        "Telegram": "org.telegram.messenger"
    }
    
    # Dropdown untuk memilih aplikasi
    apk_name = st.selectbox("Pilih Aplikasi", list(apps.keys()))
    
    # Input jumlah data yang akan diambil
    num_data = st.number_input("Jumlah Data", min_value=10, max_value=5000, step=10, value=100)

    if st.button("Submit"):
        st.write(f"Scraping ulasan {apk_name}...")

        # Nama paket aplikasi yang dipilih
        package_name = apps[apk_name]

        # Scrape ulasan dari Google Play Store secara iteratif
        total_reviews = []
        batch_size = 100  # Google Play Scraper membatasi pengambilan 100 ulasan per panggilan
        for i in range(0, num_data, batch_size):
            try:
                reviews_data, _ = reviews(
                    package_name,  # Nama paket aplikasi
                    lang='en',  # Bahasa yang digunakan adalah Inggris
                    country='us',  # Negara diatur ke Amerika Serikat
                    sort=Sort.MOST_RELEVANT,
                    count=min(batch_size, num_data - len(total_reviews)),  # Sisa ulasan yang harus diambil
                    continuation_token=None if i == 0 else _  # Token untuk melanjutkan scraping
                )
                total_reviews.extend(reviews_data)
                if len(total_reviews) >= num_data:  # Berhenti jika sudah mencapai jumlah yang diminta
                    break
            except Exception as e:
                st.error(f"Terjadi kesalahan saat scraping: {e}")
                break

        # Pastikan ada data yang diambil
        if not total_reviews:
            st.write("Tidak ada ulasan yang ditemukan.")
            return

        # Konversi data menjadi DataFrame
        df = pd.DataFrame(total_reviews)

        # Pilih hanya kolom yang relevan
        if {'reviewId', 'score', 'at', 'content', 'appVersion'}.issubset(df.columns):
            df_filtered = df[['reviewId', 'score', 'at', 'content', 'appVersion']]
        else:
            st.error("Kolom yang diperlukan tidak ditemukan dalam data.")
            return

        # Tampilkan DataFrame yang sudah difilter
        st.write("Data Ulasan yang Diperoleh (Filtered):")
        st.dataframe(df_filtered)

        # Menyimpan file CSV untuk diunduh
        csv_file = "filtered_reviews.csv"
        df_filtered.to_csv(csv_file, index=False)
        with open(csv_file, "rb") as f:
            st.download_button("📥Download Filtered CSV", data=f, file_name=csv_file)

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

            # Langkah-langkah preprocessing
            steps = [
                "Cleaning",
                "Case Folding",
                "Tokenizing",
                "Normalization",
                "Removal Stopwords",
                "Lemmatization"
            ]

            # Simulasi proses preprocessing
            for i, step in enumerate(steps):
                st.write(f"• {step}...")
                time.sleep(1)  # Simulasi waktu proses
                progress_bar.progress((i + 1) / len(steps))
                
                if step == "Cleaning":
                    data['cleaned'] = data['content'].apply(cleaning)
                    st.write("Hasil Cleaning:")
                    st.write(data[['content', 'cleaned']].head())

                elif step == "Case Folding":
                    data['case_folded'] = data['cleaned'].apply(case_folding)
                    st.write("Hasil Case Folding:")
                    st.write(data[['cleaned', 'case_folded']].head())

                elif step == "Tokenizing":
                    data['tokenized'] = data['case_folded'].apply(process_tokenization)  # Menggunakan process_tokenization yang baru
                    st.write("Hasil Tokenizing:")
                    st.write(data[['case_folded', 'tokenized']].head())

                elif step == "Normalization":
                    data['normalized'] = data['tokenized'].apply(normalize)
                    st.write("Hasil Normalization:")
                    st.write(data[['tokenized', 'normalized']].head())

                elif step == "Removal Stopwords":
                    data['stopwords_removed'] = data['normalized'].apply(remove_stopwords)
                    st.write("Hasil Removal Stopwords:")
                    st.write(data[['normalized', 'stopwords_removed']].head())

                elif step == "Lemmatization":
                    data['lemmatized'] = data['stopwords_removed'].apply(lemmatize)
                    st.write("Hasil Lemmatization:")
                    st.write(data[['stopwords_removed', 'lemmatized']].head())

            st.success("Finish Preprocessing!")

             # Membuat kolom 'fixed' yang berisi hasil penggabungan kata setelah lemmatization
            data['fixed'] = data['lemmatized'].apply(lambda x: ' '.join(x))  # Menggabungkan kata hasil lemmatization
            st.write("Hasil Penggabungan Hasil Lemmatization menjadi 'fixed':")
            st.write(data[['lemmatized', 'fixed']].head())

              # Proses Labeling dengan VADER dan hitung skor compound
            st.write("Memulai proses Labeling dan perhitungan Skor Compound...")

            # Menghitung skor compound dan sentimen
            data[['compound_score', 'label']] = data['fixed'].apply(lambda x: pd.Series(get_sentiment_and_score(x)))

            # Menghapus data dengan sentimen netral
            data = data[data['label'] != 'neutral']  # Hapus baris dengan sentimen 'neutral'

            st.write("Hasil Labeling dan Skor Compound:")
            st.write(data[['fixed', 'score', 'compound_score', 'label']].head())

            # Tampilkan hasil preprocessing akhir
            st.write("Hasil Akhir Preprocessing dan Labeling:")
            st.write(data.head())

            # Simpan hasil ke CSV
            csv_file = "preprocessed_data_with_compound.csv"
            data[['score', 'fixed', 'compound_score', 'label']].to_csv(csv_file, index=False)
            with open(csv_file, "rb") as f:
                st.download_button("📥 Download Preprocessed Data", data=f, file_name=csv_file)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def visualisasi():
    st.title("📊 Visualisasi Data Sentimen")
    st.write("Unggah file CSV untuk melakukan analisis sentimen.")

    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        st.write("📄 **Data yang diunggah:**")
        st.dataframe(data.head(10))  # Menampilkan 10 baris pertama data

        if 'score' not in data.columns or 'compound_score' not in data.columns:
            st.error("Kolom 'score' atau 'compound_score' tidak ditemukan dalam file CSV!")
        else:
            # Membersihkan data
            data = data.dropna(subset=['score', 'compound_score'])
            data = data[data['label'] != 'neutral']  # Hapus data netral

            try:
                data['score'] = pd.to_numeric(data['score'])
                data['compound_score'] = pd.to_numeric(data['compound_score'])
            except ValueError:
                st.error("Kolom 'score' dan 'compound_score' harus berupa angka!")

            # Filter data positif dan negatif
            positif_data = data[data['label'] == 'positive']
            negatif_data = data[data['label'] == 'negative']

            # Hitung jumlah sentimen
            sentiment_counts = {
                "Positif 😊": len(positif_data),
                "Negatif 😡": len(negatif_data)
            }

            # Tabel kata-kata positif dan negatif
            st.subheader("🔍 **Kata-Kata Positif dan Negatif**")
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Kata Positif**")
                if not positif_data.empty:
                    positive_words = " ".join(positif_data['fixed'])  # Kolom 'fixed' berisi teks
                    st.write(positive_words.split()[:20])  # Tampilkan 20 kata pertama
                else:
                    st.write("Tidak ada data positif.")

            with col2:
                st.write("**Kata Negatif**")
                if not negatif_data.empty:
                    negative_words = " ".join(negatif_data['fixed'])  # Kolom 'fixed' berisi teks
                    st.write(negative_words.split()[:20])  # Tampilkan 20 kata pertama
                else:
                    st.write("Tidak ada data negatif.")

            # Visualisasi Distribusi Skor Compound
            st.subheader("📊 **Distribusi Skor Compound**")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(data['compound_score'], bins=10, color='skyblue', alpha=0.7, edgecolor='black')
            ax.set_title("Distribusi Skor Compound", fontsize=16)
            ax.set_xlabel("Compound Score", fontsize=12)
            ax.set_ylabel("Frekuensi", fontsize=12)
            st.pyplot(fig)

            # Visualisasi Distribusi Sentimen
            st.subheader("📈 **Distribusi Sentimen**")
            col3, col4 = st.columns(2)

            with col3:
                st.write("**Diagram Lingkaran**")
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.pie(
                    sentiment_counts.values(),
                    labels=sentiment_counts.keys(),
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=['green', 'red']
                )
                ax.axis('equal')
                st.pyplot(fig)

            with col4:
                st.write("**Diagram Batang**")
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(sentiment_counts.keys(), sentiment_counts.values(), color=['green', 'red'])
                ax.set_ylabel("Jumlah", fontsize=12)
                ax.set_title("Distribusi Sentimen", fontsize=16)
                st.pyplot(fig)

            # Wordcloud untuk Sentimen
            st.subheader("🌟 **Visualisasi Wordcloud**")
            col5, col6 = st.columns(2)

            with col5:
                st.write("**Wordcloud Sentimen Positif**")
                if not positif_data.empty:
                    wordcloud_positif = WordCloud(
                        width=800, height=400, background_color="white"
                    ).generate(" ".join(positif_data['fixed']))
                    plt.figure(figsize=(10, 5))
                    plt.imshow(wordcloud_positif, interpolation="bilinear")
                    plt.axis("off")
                    st.pyplot(plt)

            with col6:
                st.write("**Wordcloud Sentimen Negatif**")
                if not negatif_data.empty:
                    wordcloud_negatif = WordCloud(
                        width=800, height=400, background_color="white"
                    ).generate(" ".join(negatif_data['fixed']))
                    plt.figure(figsize=(10, 5))
                    plt.imshow(wordcloud_negatif, interpolation="bilinear")
                    plt.axis("off")
                    st.pyplot(plt)


def modeling():
    st.title("🤖 Modeling dengan SVM")
    
    uploaded_file = st.file_uploader("Upload File CSV untuk Modeling", type=["csv"])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        
        # Hapus data netral
        data = data[data['score'] != 3]  # Menghapus data netral
        
        # Definisikan fitur dan label
        X = data['fixed']  # Pastikan 'fixed' adalah kolom yang sesuai
        y = data['label']  # Pastikan 'label' adalah kolom yang sesuai
        
        # Konversi teks menjadi representasi numerik menggunakan TF-IDF
        vectorizer = TfidfVectorizer()
        X_vectorized = vectorizer.fit_transform(X)

        # Pisahkan data menjadi set pelatihan dan pengujian
        X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

        # SMOTE untuk oversampling
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        # Tuning Parameter
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto'],  # Hanya relevan untuk 'rbf' dan 'poly'
            'class_weight': [None, 'balanced']
        }

        # Grid Search
        grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train_resampled, y_train_resampled)

        # Model terbaik
        best_model = grid_search.best_estimator_
        st.write(f"Best Parameters: {grid_search.best_params_}")

        # Prediksi dengan model terbaik
        y_pred_best = best_model.predict(X_test)

        # Evaluasi
        cm = confusion_matrix(y_test, y_pred_best)
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_).plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix - Best SVM")
        st.pyplot(plt)

        st.write("Classification report dengan parameter terbaik:")
        st.text(classification_report(y_test, y_pred_best))

# prediksi sentimen
def prediksi_sentimen():
    st.title("🔮Prediksi Sentimen Ulasan")
    st.write("""
        Masukkan komentar atau ulasan yang ingin Anda analisis sentimennya.
        Kami akan memberikan label sentimen (Positif, Netral, atau Negatif).
    """)
    
    user_review = st.text_area("Masukkan Ulasan", "")
    
    if st.button("Prediksi Sentimen") and user_review:
        analyzer = SentimentIntensityAnalyzer()
        sentiment_score = analyzer.polarity_scores(user_review)
        
        # Ambil score komposit (compound) dari analisis VADER
        compound_score = sentiment_score['compound']

        # Tentukan ambang batas untuk label sentimen
        if compound_score >= 0.05:  # Ambang batas untuk positif
            st.write("Sentimen: Positif")
        elif compound_score <= -0.05:  # Ambang batas untuk negatif
            st.write("Sentimen: Negatif")
        else:  # Ambang batas untuk netral
            st.write("Sentimen: Netral")

        # Menampilkan nilai polaritas
        st.write(f"Polaritas Sentimen (Score Compound): {compound_score}")

# Main function untuk mengintegrasikan dengan halaman
def main():
    pages = {
        "Home": home,
        "Scraping Data": scraping,
        "Preprocessing & Labeling": preprocessing,
        "Visualisasi Sentimen": visualisasi,  # Tetap di sini
        "Modeling SVM": modeling,  # Pindahkan ke sini
        "Prediksi Sentimen": prediksi_sentimen,
    }

    page = st.sidebar.radio("Pilih Halaman", list(pages.keys()))
    pages[page]()

if __name__ == "__main__":
    main()
