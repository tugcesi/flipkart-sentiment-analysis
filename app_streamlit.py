import os
import joblib
import numpy as np
import nltk
import streamlit as st
from nltk.corpus import stopwords
from textblob import TextBlob

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
stop_words = set(stopwords.words("english"))


# NOTE: This function name must match the one used when the vectorizer was trained and saved.
# Renaming it would break joblib deserialization of vectorizer.joblib.
def ekkok(text):
    words = TextBlob(text).words
    return [word.lemmatize() for word in words if word.lower() not in stop_words]

st.set_page_config(
    page_title="Flipkart Duygu Analizi",
    page_icon="🛒",
    layout="centered",
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "sentiment_model.joblib")
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), "vectorizer.joblib")

LABELS = {0: "Negatif 😠", 1: "Nötr 😐", 2: "Pozitif 😊"}
LABEL_COLORS = {0: "🔴", 1: "🟡", 2: "🟢"}


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def predict(text: str, model, vectorizer):
    vec = vectorizer.transform([text])
    prediction = int(model.predict(vec)[0])
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vec)[0]
    else:
        decision = model.decision_function(vec)[0]
        exp = np.exp(decision - np.max(decision))
        proba = exp / exp.sum()
    return prediction, proba


def display_result(prediction: int, proba: np.ndarray):
    label = LABELS[prediction]
    color = LABEL_COLORS[prediction]

    st.markdown("---")
    st.subheader("📊 Analiz Sonucu")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric(label="Tahmin", value=f"{color} {label}")
    with col2:
        confidence = float(proba[prediction]) * 100
        st.metric(label="Güven Skoru", value=f"{confidence:.1f}%")

    st.markdown("**Tüm Sınıflar için Güven Skorları:**")
    for cls in range(3):
        score = float(proba[cls])
        st.write(f"{LABEL_COLORS[cls]} **{LABELS[cls]}**")
        st.progress(score, text=f"{score * 100:.1f}%")


def main():
    st.title("🛒 Flipkart Ürün İncelemesi Duygu Analizi")
    st.markdown(
        """
        Bu uygulama, Flipkart ürün incelemelerinin duygusunu analiz eder ve
        yorumun **Negatif**, **Nötr** veya **Pozitif** olduğunu tahmin eder.
        """
    )

    try:
        model, vectorizer = load_model()
    except Exception as e:
        st.error(
            f"Model yüklenemedi: {e}\n\n"
            "`sentiment_model.joblib` ve `vectorizer.joblib` dosyalarının "
            "uygulama ile aynı dizinde olduğundan emin olun."
        )
        st.stop()

    tab_single, tab_multi = st.tabs(["Tek Yorum Analizi", "Çoklu Yorum Analizi"])

    with tab_single:
        st.subheader("Tek Yorum Analizi")
        review_text = st.text_area(
            "Ürün yorumunu buraya girin:",
            placeholder="Örnek: This product is amazing! Very good quality.",
            height=150,
            key="single_review",
        )

        if st.button("🔍 Analiz Et", key="analyze_single"):
            if not review_text.strip():
                st.warning("⚠️ Lütfen analiz etmek için bir yorum girin.")
            else:
                with st.spinner("Analiz ediliyor..."):
                    try:
                        prediction, proba = predict(review_text, model, vectorizer)
                        display_result(prediction, proba)
                    except Exception as e:
                        st.error(f"Analiz sırasında hata oluştu: {e}")

    with tab_multi:
        st.subheader("Çoklu Yorum Analizi")
        st.markdown("Her satıra bir yorum girin:")
        multi_text = st.text_area(
            "Yorumları girin (her satırda bir yorum):",
            placeholder=(
                "Örnek:\nGreat product!\nNot satisfied with the quality.\nIt's okay, nothing special."
            ),
            height=200,
            key="multi_review",
        )

        if st.button("🔍 Tümünü Analiz Et", key="analyze_multi"):
            reviews = [r.strip() for r in multi_text.splitlines() if r.strip()]
            if not reviews:
                st.warning("⚠️ Lütfen en az bir yorum girin.")
            else:
                with st.spinner("Yorumlar analiz ediliyor..."):
                    try:
                        results = []
                        for review in reviews:
                            prediction, proba = predict(review, model, vectorizer)
                            results.append(
                                {
                                    "Yorum": review,
                                    "Duygu": LABELS[prediction],
                                    "Negatif (%)": f"{proba[0] * 100:.1f}",
                                    "Nötr (%)": f"{proba[1] * 100:.1f}",
                                    "Pozitif (%)": f"{proba[2] * 100:.1f}",
                                }
                            )

                        st.success(f"✅ {len(results)} yorum başarıyla analiz edildi.")
                        st.markdown("---")
                        for i, res in enumerate(results, 1):
                            with st.expander(f"Yorum {i}: {res['Yorum'][:60]}..."):
                                st.write(f"**Duygu:** {res['Duygu']}")
                                st.write(f"🔴 Negatif: {res['Negatif (%)']}%")
                                st.write(f"🟡 Nötr: {res['Nötr (%)']}%")
                                st.write(f"🟢 Pozitif: {res['Pozitif (%)']}%")

                    except Exception as e:
                        st.error(f"Analiz sırasında hata oluştu: {e}")

    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: grey;'>Flipkart Duygu Analizi Uygulaması</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
