# Import necessary libraries
import torch
from torchvision import transforms, datasets
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np

# Initialize Flask app
app = Flask(__name__)
# Load the trained model for prediction
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.jit.load('model_0.pt', map_location=device)
model.eval()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Define the transformation for input images
transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor()
])

# Define the route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Define the route for image prediction
@app.route('/', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded file
        file = request.files['file']
        img = Image.open(file)
        if img.mode == "RGBA":
            img = img.convert("RGB")
        print(img.mode)
        img = transform(img)
        img = img.unsqueeze(0)

        # Perform prediction
        with torch.no_grad():
            output = model(img)
            _, predicted = torch.max(output, 1)
            predicted_class = predicted.item()
            class_name = class_names[predicted_class]

        return render_template('index.html', class_name=class_name)

if __name__ == '__main__':
    app.run(debug=True)
