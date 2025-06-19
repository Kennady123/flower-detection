from flask import Flask, render_template, request, redirect, url_for, session, flash
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
import os
import joblib
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "default_secret_key")

# Connect to MongoDB
mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(mongo_uri)
db = client["lung_cancer_db"]
users = db["users"]

# Load ML model and encoder
base_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_path, "lung_cancer_risk_model.pkl")
encoder_path = os.path.join(base_path, "label_encoders.pkl")

try:
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    print("✅ Model and Encoder loaded successfully")
except Exception as e:
    print("❌ Error loading model:", e)
    model, encoder = None, None

# ---------------------------------------
# ROUTES
# ---------------------------------------

@app.route("/")
def home():
    if "user" in session:
        return redirect(url_for("predict"))
    return redirect(url_for("signup"))  # ✅ Show signup page first

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"].lower()
        password = request.form["password"]
        hashed_pw = generate_password_hash(password)

        # Check if user already exists
        if users.find_one({"email": email}):
            flash("❌ Email already registered. Please log in.", "danger")
            return redirect(url_for("login"))

        # Insert user into MongoDB
        user_data = {"name": name, "email": email, "password": hashed_pw}
        users.insert_one(user_data)
        print("✅ User stored in MongoDB:", user_data)  # Debugging statement

        flash("✅ Signup successful! Please login.", "success")
        return redirect(url_for("login"))

    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"].lower()
        password = request.form["password"]

        user = users.find_one({"email": email})

        if not user:
            flash("❌ Email not found. Please sign up.", "danger")
            return redirect(url_for("signup"))

        if not check_password_hash(user["password"], password):
            flash("❌ Incorrect password.", "danger")
            return redirect(url_for("login"))

        session["user"] = user["name"]
        flash("✅ Login successful!", "success")
        return redirect(url_for("predict"))

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("✅ Logged out successfully.", "success")
    return redirect(url_for("login"))

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if "user" not in session:
        flash("⚠️ You need to log in first.", "warning")
        return redirect(url_for("login"))

    prediction, error = None, None

    if request.method == "POST":
        try:
            age = int(request.form.get("age", 0))
            smoke = 1 if request.form.get("smoke", "").lower() == "yes" else 0
            yellow_fingers = 1 if request.form.get("yellow_fingers", "").lower() == "yes" else 0
            anxiety = 1 if request.form.get("anxiety", "").lower() == "yes" else 0
            peer_pressure = 1 if request.form.get("peer_pressure", "").lower() == "yes" else 0
            chronic_disease = 1 if request.form.get("chronic_disease", "").lower() == "yes" else 0
            fatigue = 1 if request.form.get("fatigue", "").lower() == "yes" else 0
            allergies = 1 if request.form.get("allergies", "").lower() == "yes" else 0
            wheeze = 1 if request.form.get("wheeze", "").lower() == "yes" else 0

            sample_input = np.array([[age, smoke, yellow_fingers, anxiety, peer_pressure, chronic_disease, fatigue, allergies, wheeze]])
            print("📊 Input Data:", sample_input)

            if model and encoder:
                prediction = encoder.inverse_transform(model.predict(sample_input))[0]
            else:
                error = "❌ Model is not loaded."

        except ValueError as ve:
            error = "❌ Invalid input: " + str(ve)
        except Exception as e:
            error = "❌ Error processing request: " + str(e)

    return render_template("index.html", user=session["user"], prediction=prediction, error=error)

if __name__ == "__main__":
    app.run(debug=True)
