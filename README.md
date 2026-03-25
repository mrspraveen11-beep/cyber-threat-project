# Cyber Threat Prediction

This repository contains a Streamlit machine learning app for phishing detection.

## Run locally

1. Activate environment:
   - `venv\Scripts\Activate.ps1`
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Run app:
   - `streamlit run main.py`

## GitHub Pages

- A static page is provided in `docs/index.html` so GitHub Pages can serve content and avoid 404.
- In GitHub repo settings, choose Pages source: `main/master` branch, `/docs` folder.

## Large file note

`phishing.csv` is 54 MB, over GitHub's recommended 50 MB. Use Git LFS if needed:
- `git lfs install`
- `git lfs track "*.csv"`
- `git add .gitattributes` then commit & push.
