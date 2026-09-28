# Pneumonia Detection from Chest X-rays (Streamlit app)

Upload a chest X-ray (DICOM, PNG or JPG) and get the predicted class and pneumonia probability.
Decision-support prototype - not a medical device; a radiologist must confirm every result.

Model: **ResNet50 + extra layers (fine-tuned)** (test ROC-AUC 0.86, decision threshold 0.19).

## Run with Docker
```
docker build -t pneumonia-detector .
docker run -p 8501:8501 pneumonia-detector
```
## Run without Docker (e.g. in GitHub Codespaces)
```
pip install -r requirements.txt
python3 -m streamlit run app.py
```
Open the forwarded port 8501 (set its visibility to Public in the Ports tab).
