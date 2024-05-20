# Import necessary libraries
import torch
from torchvision import transforms, datasets, models, io
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np

# Initialize Flask app
app = Flask(__name__)
# Load the trained model for prediction
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.jit.load('pretrained_model2.pt', map_location=device)
model.eval()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Define the transformation for input images
weights = models.EfficientNet_B1_Weights.DEFAULT
transform = transforms.Compose([
    transforms.Resize(size=(32, 32)),
    weights.transforms()
])

# Define the transformation to replicate the given configuration
# transform = transforms.Compose([
#     transforms.Resize(size=(32, 32), interpolation=2),  # Bilinear interpolation is equivalent to value 2
#     transforms.CenterCrop(size=240),  # Center crop to size 240
#     transforms.Resize(size=255, interpolation=2),  # Resize to size 255 with bilinear interpolation
#     transforms.ToTensor(),  # Convert the image to a tensor
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the tensor
# ])


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
        # print(img.__dir__())
        # img = io.read_image(str(img)).type(torch.float32)
        if img.mode == "RGBA":
            img = img.convert("RGB")
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
