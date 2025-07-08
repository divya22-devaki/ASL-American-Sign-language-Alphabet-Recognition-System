import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, request, render_template, redirect, url_for
from keras.models import load_model

# Load the model
model = load_model(r'asl_vgg16_best_weights.keras', compile=False)

# Initialize the Flask app
app = Flask(__name__)

# Define route for the index page
@app.route('/')
@app.route('/index.html')
def index():
    return render_template('index.html')

# Define route for the prediction page
@app.route('/prediction.html')
def prediction():
    return render_template('prediction.html')

# Define route for the contact page
@app.route('/contact.html')
def contact():
    return render_template('contact.html')

# Define route for the about page
@app.route('/about.html')
def about():
    return render_template('about.html')

# Updated image preprocessing function to handle grayscale and RGB images
def preprocess_image(image):
    img = Image.open(image)
    
    # Convert grayscale to RGB by duplicating channels
    if img.mode == 'L':  # Grayscale
        img = img.convert("RGB")
        
    # Handle images with alpha channels by converting them to RGB
    elif img.mode == 'RGBA':
        img = img.convert("RGB")
        
    # Resize and normalize
    img = img.resize((64, 64))  # Resize to match model input
    img = np.array(img) / 255.0  # Normalize to [0, 1]
    return img

# Define route for handling prediction results
@app.route('/result', methods=['POST'])
def res():
    if request.method == 'POST':
        f = request.files['image']
        
        # Labels for ASL predictions
        labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ['del', 'nothing', 'space']
        
        # Process and predict
        img = preprocess_image(f)  # Preprocess without saving
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        predictions = model.predict(img)
        predicted_class = labels[np.argmax(predictions)]
        
        # Redirect to logout.html with the prediction as a query parameter
        return redirect(url_for('logout', pred=predicted_class))

# Update logout route to show prediction
@app.route('/logout.html')
def logout():
    predicted_class = request.args.get('pred', None)  # Get prediction from query parameter
    return render_template('logout.html', pred=predicted_class)  # Show prediction in logout.html

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
