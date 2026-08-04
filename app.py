"""
Streamlit demo: upload a photo or use the webcam, detect a face with OpenCV's
bundled Haar cascade, crop to 48x48 grayscale, and classify with the
from-scratch CNN. Shows a confidence bar for all 7 classes, not just the
top-1 label, and a rolling average in webcam mode to avoid frame-to-frame
flicker.
"""
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image

from model import EmotionCNN

ROOT = Path(__file__).parent
CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

FACE_CASCADE = cv2.CascadeClassifier(str(ROOT / "models" / "haarcascade_frontalface_default.xml"))


@st.cache_resource
def load_model():
    model = EmotionCNN().to(DEVICE)
    model.load_state_dict(torch.load(ROOT / "best_model.pt", map_location=DEVICE))
    model.eval()
    return model


def predict(model, face_gray_48):
    x = face_gray_48.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    tensor = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
    return probs


def detect_and_crop(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    if len(faces) == 0:
        return None, None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face = cv2.resize(gray[y:y + h, x:x + w], (48, 48))
    return face, (x, y, w, h)


def draw_bars(probs):
    for cls, p in sorted(zip(CLASSES, probs), key=lambda t: -t[1]):
        st.write(f"**{cls}**")
        st.progress(float(p))
        st.caption(f"{p*100:.1f}%")


def main():
    st.set_page_config(page_title="Facial Emotion Recognition", layout="centered")
    st.title("Facial Emotion Recognition")
    st.caption("Trained from scratch on FER-2013 (35,887 images) — a noisy, real-world dataset. "
               "Published baselines land around 65-70% test accuracy; see README for this run's actual numbers.")

    model = load_model()
    mode = st.radio("Input", ["Upload photo", "Webcam snapshot", "Live webcam (rolling average)"])

    if mode == "Upload photo":
        uploaded = st.file_uploader("Upload a face photo", type=["jpg", "jpeg", "png"])
        if uploaded:
            img = np.array(Image.open(uploaded).convert("RGB"))[:, :, ::-1]
            face, box = detect_and_crop(img)
            st.image(img[:, :, ::-1], caption="input", width=300)
            if face is None:
                st.warning("No face detected.")
            else:
                probs = predict(model, face)
                st.image(face, caption="detected face (48x48)", width=150)
                draw_bars(probs)

    elif mode == "Webcam snapshot":
        shot = st.camera_input("Take a photo")
        if shot:
            img = np.array(Image.open(shot).convert("RGB"))[:, :, ::-1]
            face, box = detect_and_crop(img)
            if face is None:
                st.warning("No face detected.")
            else:
                probs = predict(model, face)
                st.image(face, caption="detected face (48x48)", width=150)
                draw_bars(probs)

    else:
        st.info("Streams webcam frames through OpenCV directly (outside the Streamlit request loop) "
                "and averages predictions over the last 10 frames to smooth flicker. Close the window to stop.")
        if st.button("Start live webcam"):
            cap = cv2.VideoCapture(0)
            window = deque(maxlen=10)
            placeholder = st.empty()
            stop = st.button("Stop")
            while cap.isOpened() and not stop:
                ok, frame = cap.read()
                if not ok:
                    break
                face, box = detect_and_crop(frame)
                if face is not None:
                    probs = predict(model, face)
                    window.append(probs)
                    avg = np.mean(window, axis=0)
                    x, y, w, h = box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    label = f"{CLASSES[int(np.argmax(avg))]} ({avg.max()*100:.0f}%)"
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                placeholder.image(frame[:, :, ::-1])
            cap.release()


if __name__ == "__main__":
    main()
