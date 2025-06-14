# Ideation-Detection

A lightweight PyTorch + Hugging Face project to detect suicidal ideation in text using a fine-tuned DistilBERT model.

```
python3 -m venv venv
source venv/bin/activate
```

```
pip install --upgrade pip
pip install -r requirements.txt
```

Ran on Google Colab with T4 GPU runtime
```
python model_training.py
python evaluate_model.py
```

Save the model locally from Google Colab
```
!zip -r results.zip results
from google.colab import files
files.download("results.zip")
```
