# Deployment Guide

## Deploy on Streamlit Community Cloud (Recommended)

### Step 1: Go to Streamlit Cloud
1. Visit https://share.streamlit.io
2. Sign up or log in with your GitHub account

### Step 2: Create New App
1. Click **"New app"** button
2. Select repository: `mrspraveen11-beep/cyber-threat-project`
3. Select branch: `master` (or `main`)
4. Select main file: `app.py`
5. Click **Deploy**

### Step 3: Wait for Deployment
- Streamlit will build and deploy your app
- You'll get a URL like: `https://cyber-threat-project-<random>.streamlit.app`
- Initial deployment takes 2-3 minutes

### Step 4: Share & Use
- Share your public URL with anyone
- App will auto-update when you push to GitHub
- Free tier includes ~1GB storage and ~5 private apps

---

## Local Deployment (Docker)

If you want to deploy on your own server or cloud:

### Build Docker Image
```bash
docker build -t cyber-threat .
```

### Create Dockerfile
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

### Run Locally with Docker
```bash
docker run -p 8501:8501 cyber-threat
```

---

## Alternative Cloud Platforms

### Heroku
```bash
heroku create cyber-threat-prediction
heroku buildpacks:set heroku/python
git push heroku master
```

### Render / Railway / Hugging Face Spaces
- Similar process with connection to GitHub
- Usually simpler than Heroku
- Recommended: Render (https://render.com) or Railway (https://railway.app)

---

## Troubleshooting

**Issue: Model not found**
- Ensure `phishing_model.pkl`, `scaler.pkl`, `feature_columns.pkl` are in repo
- Train model locally first: `python -c "from app import train_model; train_model(pd.read_csv('phishing.csv'))"`

**Issue: Large file (phishing.csv)**
- GitHub warns about 54MB CSV file
- Option 1: Use Git LFS (recommended)
- Option 2: Accept warning and proceed
- Option 3: Store CSV externally (AWS S3, etc.) and fetch at runtime

**Issue: Slow prediction**
- Web scraping has 5s timeout
- Normal URLs take 2-5 seconds to analyze
- Consider adding caching or offline mode for production

---

## Monitoring

After deployment:
- Check app logs in Streamlit Cloud dashboard
- Monitor memory usage (free tier: ~1GB)
- Monitor CPU usage for prediction timeouts
- Set up error alerts if needed
