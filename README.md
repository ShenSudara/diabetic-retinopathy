# Diabetic Retinopathy Severity Classifier

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://diabetic-retinopathy-2gzvgjqoqu5nh49nupmqfg.streamlit.app)

A web app that grades the severity of diabetic retinopathy (DR) from a retinal fundus
photograph. Upload a JPEG, choose **EfficientNetB0** or **DenseNet121**, and the app
returns a 5-class prediction with per-class probabilities and a **Grad-CAM** heatmap
showing which regions of the retina influenced the decision.

**Live demo:** <https://diabetic-retinopathy-2gzvgjqoqu5nh49nupmqfg.streamlit.app>

> **Disclaimer:** For research and education only. This is not a medical device and its
> output is not a diagnosis. Consult a qualified eye-care professional.

---

## Screenshots

<!-- Replace the paths below with your own screenshots, e.g. docs/screenshots/home.png -->

| Upload & settings | Prediction result |
|---|---|
| ![Upload screen](docs/screenshots/home.png) | ![Prediction result](docs/screenshots/result.png) |

**Grad-CAM visualisation**

![Grad-CAM heatmap](docs/screenshots/gradcam.png)

---

## About diabetic retinopathy

Diabetic retinopathy is a complication of diabetes in which high blood sugar damages
the small blood vessels of the retina. It is one of the leading causes of preventable
vision loss in working-age adults. Because early stages often have no symptoms, regular
retinal screening is important, and automated grading can help prioritise patients who
need specialist review.

The app classifies images into the five standard severity grades:

| Grade | Class | Description |
|---|---|---|
| 0 | No DR | No visible signs of retinopathy |
| 1 | Mild | Microaneurysms only |
| 2 | Moderate | More than microaneurysms, but less than severe |
| 3 | Severe | Extensive haemorrhages / venous beading, no new vessel growth |
| 4 | Proliferative DR | Abnormal new blood vessels (neovascularisation) |

## Features

- **Two models:** fine-tuned EfficientNetB0 and DenseNet121 (ImageNet transfer
  learning), selectable from the sidebar.
- **Per-class probabilities** shown in clinical severity order, plus a low-confidence
  warning when the top class is below 50%.
- **Grad-CAM explainability:** a heatmap overlay of the regions that drove the prediction.
- **Same preprocessing as training:** you can see both the resized input and the
  preprocessed image the model actually receives.
- **Upload validation:** rejects non-JPEG, renamed, truncated, corrupted, oversized
  (>20 MB) and tiny (<64 px) files with a clear message.
- **Low memory use:** only one model is held in memory at a time, so the app fits on
  the Streamlit Community Cloud free tier.

## How it works

```
JPEG upload
  └─> validation (signature, format, full decode, size)
  └─> decode (tf.io.decode_jpeg) ─> resize to 224×224 (bilinear)
  └─> brightness normalisation (target mean 75)
  └─> contrast enhancement (CLAHE on the L channel of LAB)
  └─> noise reduction (bilateral filter)
  └─> sharpening (unsharp mask)
  └─> CNN (EfficientNetB0 / DenseNet121) ─> softmax over 5 classes
  └─> Grad-CAM on the backbone's final feature map
```

### Model architecture

Both models share the same classification head on top of a pretrained backbone:

```
Backbone (EfficientNetB0 / DenseNet121)
  ├─> GlobalAveragePooling2D ─┐
  └─> GlobalMaxPooling2D ─────┴─> Concatenate
        ─> BatchNorm ─> Dropout ─> Dense(512)
        ─> BatchNorm ─> Dropout ─> Dense(5, softmax)
```

Grad-CAM uses the backbone's output feature map, which is the input of the pooling
layers. It weights each channel by its average gradient with respect to the predicted
class score and overlays the result on the input image.

## Tech stack

- [TensorFlow / Keras](https://www.tensorflow.org/) for the models and inference
- [OpenCV](https://opencv.org/) for image preprocessing and the heatmap overlay
- [Streamlit](https://streamlit.io/) for the web UI, deployed on Streamlit Community Cloud
- Pillow and NumPy

## Project structure

```
diabetic-retinopathy/
├── app.py              # Streamlit UI
├── preprocessing.py    # upload validation + training-time OpenCV preprocessing
├── inference.py        # model loading, prediction, Grad-CAM
├── models/
│   ├── EfficientNetB0_best_model_fine_tuned_1.keras
│   └── DenseNet121_best_model_fine_tuned_1.keras
├── requirements.txt
└── .streamlit/config.toml
```

## Run locally

Requires Python 3.11 or 3.12.

```bash
git clone https://github.com/ShenSudara/diabetic-retinopathy.git
cd diabetic-retinopathy
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then open <http://localhost:8501>.

## Deployment

The app is deployed on [Streamlit Community Cloud](https://share.streamlit.io) from the
`main` branch with `app.py` as the entry point. The model files (about 97 MB and 60 MB)
are committed directly to the repository, below GitHub's 100 MB per-file limit.

### Troubleshooting

- **"Could not load ..."**: the TensorFlow/Keras version differs from the one used for
  training. Pin TensorFlow in `requirements.txt`.
- **App restarts / memory error**: the free tier ran out of RAM. Only one model should be
  loaded at a time.
- **`ImportError: libGL.so.1`**: install `opencv-python-headless`, not `opencv-python`.
- **Grad-CAM unavailable**: the prediction still works; the message in the app explains why.

## Limitations

- Trained on 224×224 fundus images; photos from other cameras, very different lighting or
  non-fundus images can give unreliable results.
- The model sees only the image, with no clinical history.
- Grad-CAM shows where the model looked, not a verified lesion segmentation.

## Acknowledgements

Developed as a Computer Vision project at NIBM.
