# Diabetic Retinopathy Severity Classifier (Streamlit)

Upload a retinal fundus JPEG, pick **EfficientNetB0** or **DenseNet121**, and get a
5-class severity prediction (No DR, Mild, Moderate, Severe, Proliferative DR) with a
Grad-CAM heatmap.

> For research and education only. Not a medical device; not a diagnosis.

## Project layout

```
dr-app/
├── app.py              # Streamlit UI
├── preprocessing.py    # JPEG/corruption checks + your OpenCV preprocessing
├── inference.py        # model loading, prediction, Grad-CAM
├── models/             # <- put your two .keras files here
│   ├── EfficientNetB0_best_model_fine_tuned_1.keras
│   └── DenseNet121_best_model_fine_tuned_1.keras
├── requirements.txt
└── .streamlit/config.toml
```

## 1. Before anything else: pin your TensorFlow version

Open `requirements.txt` and replace the TensorFlow line with the version you trained
with (in your training environment: `pip show tensorflow`). Pick the **same Python
minor version** too (3.11 or 3.12 is a safe choice for TF 2.16 - 2.2x).

## 2. Run locally

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## 3. Check predictions match your notebook (do this!)

Take 3-5 images you already have predictions for in the notebook, upload them, and
compare the predicted class and probabilities. They should agree to roughly 2-3
decimal places. If they do not, the usual causes are:

| Symptom | Likely cause |
|---|---|
| Always the same class | Resize method or colour order differs from training |
| Tiny differences (4th decimal or later) | Different TensorFlow build / CPU floating-point (harmless) |
| Model fails to load | TensorFlow/Keras version differs from training |

Details of what the app assumes about training:

- Images were decoded with `tf.io.decode_jpeg` and resized with `tf.image.resize`
  (bilinear) to 224x224 **before** the OpenCV steps. The app does exactly this, and was
  verified to give bit-identical pixels to that path.
- The array went into the OpenCV functions in RGB order with **no conversion**, so the
  app does the same (the functions treat it as BGR; that is how the model was trained).
- No `/255` scaling and no `preprocess_input` is applied afterwards.

If any of these is wrong for your notebook, the only place to change is `prepare_image`
(and `_decode_like_training`) in `preprocessing.py`.

## 4. Deploy on Streamlit Community Cloud

1. Create a GitHub repository and push this folder, **including the `models/` folder**.
   Your files (97 MB and 60 MB) are under GitHub's 100 MB hard limit, so normal
   `git add` works. GitHub will print a warning for files over 50 MB; that is fine.
   ```bash
   git init
   git add .
   git commit -m "DR classifier app"
   git branch -M main
   git remote add origin https://github.com/<you>/<repo>.git
   git push -u origin main
   ```
   If the push is rejected for size, use Git LFS (`git lfs track "*.keras"`) or host the
   models on Hugging Face Hub and download them at startup.
2. Go to <https://share.streamlit.io>, sign in with GitHub, click **Create app**.
3. Choose the repository, branch `main`, main file `app.py`.
4. Open **Advanced settings** and select Python **3.11 or 3.12** (match training).
5. Click **Deploy**. The first build installs TensorFlow, which takes a few minutes.

### Memory

The app keeps **only one model in memory at a time** (switching models unloads the
previous one) so it stays inside the free-tier memory limit. The first prediction after
switching models is slower because the model has to load.

### Troubleshooting

- **"Could not load ..."**: version mismatch. Pin TensorFlow to the training version.
- **App restarts / "Oh no" memory error**: free tier ran out of RAM. Confirm only one
  model loads at a time, or move to Hugging Face Spaces / a larger host.
- **`ImportError: libGL.so.1`**: you installed `opencv-python` instead of
  `opencv-python-headless`.
- **Grad-CAM unavailable**: the prediction still works; the message in the app shows why.
