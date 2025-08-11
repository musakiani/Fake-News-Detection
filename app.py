import os
from flask import Flask, render_template, request
import joblib
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)

# Load models - paths ko apne project structure ke mutabiq adjust karen
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "Models", "fake_news_clf.pkl"))
tfidf = joblib.load(os.path.join(BASE_DIR, "Models", "tfidf.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "Models", "scaler (1).pkl"))

mobilenet = MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=(128, 128, 3))

def extract_image_feature(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = mobilenet.predict(img_array, verbose=0)
    return features.flatten()

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_text = ""
    prediction_color = ""
    if request.method == "POST":
        text = request.form["text"]
        img = request.files["image"]

        # Image save karne se pehle folder existence check karen
        img_path = os.path.join(BASE_DIR, "static", "uploaded.jpg")
        upload_folder = os.path.dirname(img_path)
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)

        img.save(img_path)

        # Text feature extraction
        text_feature = tfidf.transform([text]).toarray()

        # Image feature extraction
        img_feature = extract_image_feature(img_path).reshape(1, -1)

        # Features combine karen aur scale karen
        combined_features = np.hstack((text_feature, img_feature))
        combined_features = scaler.transform(combined_features)

        # Prediction karen
        pred = model.predict(combined_features)[0]

        # Result set karen
        if pred == "FAKE":
            prediction_text = "Fake News ❌"
            prediction_color = "red"
        else:
            prediction_text = "Real News ✅"
            prediction_color = "green"

    return render_template("index.html", prediction=prediction_text, color=prediction_color)

if __name__ == "__main__":
    app.run(debug=True)