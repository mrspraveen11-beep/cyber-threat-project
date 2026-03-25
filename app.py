import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re
import io
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import requests
from bs4 import BeautifulSoup
import difflib
import plotly.express as px


# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Cyber Threat Prediction",
    page_icon="🛡️",
    layout="wide"
)

# ================= HEADER =================
st.markdown("""
<style>
.title {
    font-size: 2.5rem;
    font-weight: bold;
    text-align: center;
    color: #4FC3F7;
}
.subtitle {
    text-align: center;
    color: #B0BEC5;
}
.safe {
    background-color: #1B5E20;
    padding: 15px;
    border-radius: 10px;
    color: white;
    font-size: 1.2rem;
}
.danger {
    background-color: #B71C1C;
    padding: 15px;
    border-radius: 10px;
    color: white;
    font-size: 1.2rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🛡️ Cyber Threat Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Phishing Website Detection using Machine Learning</div>', unsafe_allow_html=True)

# ================= SIDEBAR =================
menu = st.sidebar.radio(
    "Menu",
    ["Home", "EDA & Visualization", "Train Model", "Predict"]
)

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    return pd.read_csv("phishing.csv")

# ================= TRAIN MODEL =================
def train_model(df):
    drop_cols = ['FILENAME', 'URL', 'Domain', 'Title']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    y = X.pop("label")

    X = X.select_dtypes(include=[np.number]).fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))

    joblib.dump(model, "phishing_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(X.columns.tolist(), "feature_columns.pkl")

    return acc

# ================= EDA & VISUALIZATION =================
if menu == "EDA & Visualization":
    st.subheader("📊 Exploratory Data Analysis (EDA)")
    
    try:
        df = load_data()
        st.success(f"Dataset loaded — {df.shape[0]:,} rows × {df.shape[1]} columns")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "Overview", 
            "Target Distribution", 
            "Feature Distributions", 
            "Correlation"
        ])
        
        with tab1:
            st.write("**First 5 rows**")
            st.dataframe(df.head())
            
            st.write("**Dataset information**")
            buffer = io.StringIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue())
            
            st.write("**Missing values**")
            missing = df.isnull().sum()
            if missing.sum() > 0:
                st.write(missing[missing > 0])
            else:
                st.success("No missing values 🎉")
            
            st.write("**Memory usage**")
            st.write(f"≈ {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        with tab2:
            st.write("**Target variable distribution (label)**")
            # 0 = Phishing, 1 = Legitimate (common in this kind of dataset)
            label_map = {0: "Phishing (0)", 1: "Legitimate (1)"}
            label_counts = df['label'].map(label_map).value_counts()
            
            col_chart, col_metrics = st.columns([3, 2])
            
            with col_chart:
                fig_pie = px.pie(
                    values=label_counts.values,
                    names=label_counts.index,
                    title="Phishing vs Legitimate Websites",
                    color_discrete_sequence=["#EF553B", "#00CC96"]
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col_metrics:
                st.metric("Total samples", f"{len(df):,}")
                st.metric("Phishing samples", f"{label_counts.get('Phishing (0)', 0):,}", delta_color="inverse")
                st.metric("Legitimate samples", f"{label_counts.get('Legitimate (1)', 0):,}")
        
        with tab3:
            st.write("**Distribution of numerical features by class**")
            
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            if 'label' in numeric_cols:
                numeric_cols.remove('label')
            
            selected_features = st.multiselect(
                "Select features to visualize",
                options=sorted(numeric_cols),
                default=['URLLength', 'DomainLength', 'NoOfSubDomain', 'URLSimilarityIndex', 'TLDLegitimateProb', 'NoOfOtherSpecialCharsInURL']
            )
            
            if selected_features:
                for col in selected_features:
                    with st.expander(f"📈 {col}", expanded=True):
                        fig_hist = px.histogram(
                            df,
                            x=col,
                            color='label',
                            barmode='overlay',
                            title=f"{col} distribution by class",
                            labels={'label': 'Class'},
                            color_discrete_map={0: "#EF553B", 1: "#00CC96"}
                        )
                        fig_hist.update_layout(bargap=0.1)
                        st.plotly_chart(fig_hist, use_container_width=True)
        
        with tab4:
            st.write("**Correlation between features**")
            
            if st.button("Show Correlation Heatmap"):
                with st.spinner("Calculating correlation..."):
                    corr = df.select_dtypes(include=['number']).corr()
                    
                    fig_heat = px.imshow(
                        corr,
                        text_auto=".2f",
                        aspect="auto",
                        color_continuous_scale="RdBu_r",
                        title="Feature Correlation Heatmap"
                    )
                    fig_heat.update_layout(height=900, width=900)
                    st.plotly_chart(fig_heat, use_container_width=True)

    except FileNotFoundError:
        st.error("phishing.csv file not found in the current directory.")
    except Exception as e:
        st.error(f"Error during EDA: {str(e)}")

# ================= URL FEATURE EXTRACTION =================
def extract_url_features(url, feature_columns):
    row = pd.DataFrame(0, columns=feature_columns, index=[0])

    parsed = urlparse(url)
    domain = parsed.netloc
    path = parsed.path + parsed.query
    scheme = parsed.scheme

    # URL-based features
    if "URLLength" in row.columns:
        row["URLLength"] = len(url)

    if "DomainLength" in row.columns:
        row["DomainLength"] = len(domain)

    if "IsDomainIP" in row.columns:
        row["IsDomainIP"] = 1 if re.match(r'^(\d{1,3}\.){3}\d{1,3}$', domain) else 0

    if "TLDLength" in row.columns:
        row["TLDLength"] = len(domain.split('.')[-1]) if '.' in domain else 0

    if "NoOfSubDomain" in row.columns:
        row["NoOfSubDomain"] = domain.count('.') - 1 if not row["IsDomainIP"].values[0] else 0

    if "HasObfuscation" in row.columns:
        row["HasObfuscation"] = 1 if any(c in url for c in ['@', '%']) else 0

    if "NoOfObfuscatedChar" in row.columns:
        row["NoOfObfuscatedChar"] = url.count('%') + url.count('@')

    if "ObfuscationRatio" in row.columns:
        row["ObfuscationRatio"] = row["NoOfObfuscatedChar"].values[0] / len(url) if len(url) else 0

    if "NoOfLettersInURL" in row.columns:
        row["NoOfLettersInURL"] = sum(c.isalpha() for c in url)

    if "LetterRatioInURL" in row.columns:
        row["LetterRatioInURL"] = row["NoOfLettersInURL"].values[0] / len(url) if len(url) else 0

    if "NoOfDegitsInURL" in row.columns:
        row["NoOfDegitsInURL"] = sum(c.isdigit() for c in url)

    if "DegitRatioInURL" in row.columns:
        row["DegitRatioInURL"] = row["NoOfDegitsInURL"].values[0] / len(url) if len(url) else 0

    if "NoOfEqualsInURL" in row.columns:
        row["NoOfEqualsInURL"] = url.count('=')

    if "NoOfQMarkInURL" in row.columns:
        row["NoOfQMarkInURL"] = url.count('?')

    if "NoOfAmpersandInURL" in row.columns:
        row["NoOfAmpersandInURL"] = url.count('&')

    if "NoOfOtherSpecialCharsInURL" in row.columns:
        special = len(re.findall(r'[^a-zA-Z0-9\.\:/\?=&]', path))
        row["NoOfOtherSpecialCharsInURL"] = special

    if "SpacialCharRatioInURL" in row.columns:
        total_special = (row["NoOfOtherSpecialCharsInURL"].values[0] + 
                        row["NoOfEqualsInURL"].values[0] + 
                        row["NoOfQMarkInURL"].values[0] + 
                        row["NoOfAmpersandInURL"].values[0])
        row["SpacialCharRatioInURL"] = total_special / len(url) if len(url) else 0

    if "IsHTTPS" in row.columns:
        row["IsHTTPS"] = 1 if scheme == 'https' else 0

    # Approximate probabilities (you can improve this later)
    if "TLDLegitimateProb" in row.columns:
        tld_probs = {'com': 0.5229, 'org': 0.0799, 'net': 0.1, 'de': 0.0326, 'uk': 0.0285, 
                     'in': 0.005, 'ru': 0.018, 'top': 0.0002, 'vn': 0.0013, 'app': 0.0015}
        tld = domain.split('.')[-1] if '.' in domain else ''
        row["TLDLegitimateProb"] = tld_probs.get(tld, 0.001)

    if "URLCharProb" in row.columns:
        row["URLCharProb"] = 0.06

    if "CharContinuationRate" in row.columns:
        continued = 0
        for i in range(1, len(url)):
            if url[i].isalnum() and url[i-1].isalnum() and (url[i].isalpha() == url[i-1].isalpha()):
                continued += 1
        row["CharContinuationRate"] = continued / (len(url) - 1) if len(url) > 1 else 0

    if "URLSimilarityIndex" in row.columns:
        has_digit = any(c.isdigit() for c in domain)
        has_dash = '-' in domain
        if not has_digit and not has_dash:
            row["URLSimilarityIndex"] = 100
        elif has_dash:
            row["URLSimilarityIndex"] = 80
        else:
            row["URLSimilarityIndex"] = 50

    # Content-based features (same as before)
    try:
        response = requests.get(url, timeout=5, allow_redirects=True)
        html = response.text
        soup = BeautifulSoup(html, 'html.parser')

        if "LineOfCode" in row.columns:
            row["LineOfCode"] = len(html.splitlines())

        if "LargestLineLength" in row.columns:
            row["LargestLineLength"] = max((len(line) for line in html.splitlines()), default=0)

        if "HasTitle" in row.columns:
            row["HasTitle"] = 1 if soup.title else 0

        title = soup.title.string if soup.title else ''

        if "DomainTitleMatchScore" in row.columns:
            row["DomainTitleMatchScore"] = difflib.SequenceMatcher(None, domain.lower(), title.lower()).ratio() * 100

        if "URLTitleMatchScore" in row.columns:
            row["URLTitleMatchScore"] = difflib.SequenceMatcher(None, url.lower(), title.lower()).ratio() * 100

        if "HasFavicon" in row.columns:
            row["HasFavicon"] = 1 if soup.find("link", rel=lambda x: x and "icon" in x.lower()) else 0

        if "Robots" in row.columns:
            try:
                robots_resp = requests.head(url.rstrip('/') + '/robots.txt', timeout=3)
                row["Robots"] = 1 if robots_resp.status_code == 200 else 0
            except:
                row["Robots"] = 0

        if "IsResponsive" in row.columns:
            row["IsResponsive"] = 1 if soup.find("meta", attrs={"name": "viewport"}) else 0

        if "NoOfURLRedirect" in row.columns:
            row["NoOfURLRedirect"] = len(response.history)

        if "NoOfSelfRedirect" in row.columns:
            row["NoOfSelfRedirect"] = sum(1 for r in response.history if domain in urlparse(r.url).netloc)

        if "HasDescription" in row.columns:
            row["HasDescription"] = 1 if soup.find("meta", attrs={"name": "description"}) else 0

        if "NoOfPopup" in row.columns:
            row["NoOfPopup"] = html.lower().count("window.open")

        if "NoOfiFrame" in row.columns:
            row["NoOfiFrame"] = len(soup.find_all("iframe"))

        if "HasExternalFormSubmit" in row.columns:
            forms = soup.find_all("form")
            row["HasExternalFormSubmit"] = 1 if any(form.get("action") and domain not in urlparse(form.get("action", "")).netloc for form in forms) else 0

        if "HasSocialNet" in row.columns:
            links = [a.get("href", "").lower() for a in soup.find_all("a", href=True)]
            social_sites = ["facebook", "twitter", "linkedin", "instagram", "youtube"]
            row["HasSocialNet"] = 1 if any(any(site in link for site in social_sites) for link in links) else 0

        if "HasSubmitButton" in row.columns:
            row["HasSubmitButton"] = 1 if soup.find("input", {"type": "submit"}) or soup.find("button", string=re.compile("submit|send|login|sign", re.I)) else 0

        if "HasHiddenFields" in row.columns:
            row["HasHiddenFields"] = 1 if soup.find("input", {"type": "hidden"}) else 0

        if "HasPasswordField" in row.columns:
            row["HasPasswordField"] = 1 if soup.find("input", {"type": "password"}) else 0

        if "Bank" in row.columns:
            row["Bank"] = 1 if "bank" in html.lower() else 0

        if "Pay" in row.columns:
            row["Pay"] = 1 if any(word in html.lower() for word in ["pay", "payment", "paypal"]) else 0

        if "Crypto" in row.columns:
            row["Crypto"] = 1 if any(word in html.lower() for word in ["crypto", "bitcoin", "wallet"]) else 0

        if "HasCopyrightInfo" in row.columns:
            row["HasCopyrightInfo"] = 1 if "copyright" in html.lower() or "©" in html else 0

        if "NoOfImage" in row.columns:
            row["NoOfImage"] = len(soup.find_all("img"))

        if "NoOfCSS" in row.columns:
            row["NoOfCSS"] = len(soup.find_all("link", rel="stylesheet"))

        if "NoOfJS" in row.columns:
            row["NoOfJS"] = len(soup.find_all("script"))

        if "NoOfSelfRef" in row.columns:
            row["NoOfSelfRef"] = sum(1 for a in soup.find_all("a", href=True) if domain in urlparse(a["href"]).netloc)

        if "NoOfEmptyRef" in row.columns:
            row["NoOfEmptyRef"] = sum(1 for a in soup.find_all("a") if not a.get("href"))

        if "NoOfExternalRef" in row.columns:
            row["NoOfExternalRef"] = sum(1 for a in soup.find_all("a", href=True) if domain not in urlparse(a["href"]).netloc)

    except Exception as e:
        st.warning(f"Could not fetch page content: {e}. Using only URL-based features.")

    return row

# ================= HOME =================
if menu == "Home":
    st.success("Application is running correctly ✅")
    st.markdown("""
    **How to use this system**
    1. Go to **EDA & Visualization** → understand your data
    2. Go to **Train Model** → create the prediction model
    3. Go to **Predict** → test any website URL
    """)

# ================= TRAIN =================
elif menu == "Train Model":
    st.subheader("🧠 Train Machine Learning Model")

    df = load_data()
    st.write("Dataset shape:", df.shape)

    if st.button("🚀 Start Training"):
        with st.spinner("Training Random Forest model..."):
            acc = train_model(df)

        st.success(f"Model trained successfully! Test accuracy: **{acc:.2%}**")
        st.write("Files created:")
        st.write("- phishing_model.pkl")
        st.write("- scaler.pkl")
        st.write("- feature_columns.pkl")

# ================= PREDICT =================
elif menu == "Predict":
    st.subheader("🔮 Check Website Safety")

    if not os.path.exists("phishing_model.pkl"):
        st.error("❌ Please train the model first in the 'Train Model' section")
        st.stop()

    model = joblib.load("phishing_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")

    url = st.text_input(
        "🌐 Enter Website URL",
        placeholder="https://www.example.com"
    )

    if st.button("Check Website"):
        if not url:
            st.warning("Please enter a URL")
        elif not url.startswith(("http://", "https://")):
            st.warning("URL should start with http:// or https://")
        else:
            try:
                with st.spinner("Analyzing URL and content..."):
                    features = extract_url_features(url, feature_columns)
                    features_scaled = scaler.transform(features)

                    prediction = model.predict(features_scaled)[0]
                    # Probability of being phishing (class 0)
                    phishing_prob = model.predict_proba(features_scaled)[0][0] * 100

                if prediction == 0:
                    st.markdown(
                        f"""
                        <div class="danger">
                            ⚠️ <b>PHISHING WEBSITE DETECTED</b><br>
                            Risk Probability: {phishing_prob:.1f}%
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="safe">
                            ✅ <b>LEGITIMATE WEBSITE</b><br>
                            Risk Probability: {phishing_prob:.1f}%
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")