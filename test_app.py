# -*- coding: utf-8 -*-
"""app.py - Flask-based Chatbot Website for Diabetic Retinopathy Diagnosis"""

import os
import argparse
from flask import Flask, render_template, request
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import timm
import logging
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Running on {DEVICE}")

# Transformations for ViT (matched with vit_novelty2.py and test.py)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create a directory for uploaded images
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ViT Model Definition (matched with vit_novelty2.py and test.py)
class Net(nn.Module):
    def __init__(self, num_classes=5):
        super(Net, self).__init__()
        self.vit = timm.create_model('vit_base_patch32_224', pretrained=False, num_classes=0)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(768),
            nn.Linear(768, 11),
            nn.ReLU(),
            nn.BatchNorm1d(11),
            nn.Linear(11, num_classes)
        )

    def forward(self, x):
        x = self.vit(x)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

# Classes from the dataset (matched with vit_novelty2.py and test.py)
CLASSES = ["Mild", "Moderate", "No_DR", "Proliferate_DR", "Severe"]

# Initialize the Grok model for report generation
MODEL = "llama-3.3-70b-versatile"
model = ChatGroq(
    temperature=0.7,
    model_name=MODEL,
    api_key="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",
    max_tokens=8000
)

# To get string output
parser = StrOutputParser()
chain = model | parser

# Prompt template for generating a medical report (unchanged)
MedicalReportTemplate = """
Generate a detailed medical report based on the following patient details and diagnosis. The report should be written in a professional tone suitable for a medical context and include the following sections:

- Overview: A summary of the patient's condition and diagnosis.
- Disease Severity: Assess the severity of the condition (e.g., mild, moderate, severe).
- Probable Causes: Identify potential causes of the condition based on the medical history.
- Recommended Medications: Suggest appropriate medications or treatments.
- Lifestyle Recommendations: Provide lifestyle changes to manage the condition.
- Follow-Up Plan: Recommend a follow-up schedule and additional tests if needed.

Ensure the report is concise, clear, and tailored to the patient's specific condition and history. Use the patient's name in the report for personalization.

Patient Details:
Name: {name}
Age: {age}
Gender: {gender}
Medical History: {medical_history}

Diagnosis:
Predicted Condition: {condition}
Confidence Score: {confidence_score:.2%}

Additional Instructions:
- For the "Predicted Condition," interpret the condition as follows:
  - "No_DR": No diabetic retinopathy, indicating no retinal damage.
  - "Mild": Mild diabetic retinopathy, early-stage retinal changes.
  - "Moderate": Moderate diabetic retinopathy, with more significant retinal changes.
  - "Severe": Severe diabetic retinopathy, with extensive retinal damage.
  - "Proliferate_DR": Proliferative diabetic retinopathy, the most advanced stage with new blood vessel growth.
- Use the medical history to inform the probable causes and recommendations.
- Give each and every detail in consised manner
- Highlight the medical terms if specified with bold and itallic format
- Include a note at the end stating: "Note: This report is AI-generated and should be validated by a medical professional."
"""

MedicalReportPrompt = PromptTemplate.from_template(MedicalReportTemplate)
MedicalReportChain = (MedicalReportPrompt | model | parser)

# Prediction Function (aligned with test.py)
def predict_image(image, model_path):
    try:
        # Initialize model per prediction
        net = Net().to(DEVICE)
        if not os.path.exists(model_path):
            logger.error(f"Model file '{model_path}' not found.")
            return None, None, None
        net.load_state_dict(torch.load(model_path, map_location=DEVICE))
        logger.info(f"Loaded model weights from {model_path}")
        net.eval()

        # Process image
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)
        logger.info("Image processed successfully")

        # Predict
        with torch.no_grad():
            output = net(image_tensor)
            if torch.isnan(output).any() or torch.isinf(output).any():
                logger.error("NaN/Inf in model output")
                return None, None, None
            probs = torch.exp(output)
            confidence, predicted = torch.max(probs, 1)
            prediction = predicted.item()
            conf_score = confidence.item()
            predicted_label = CLASSES[prediction]

        return predicted_label, conf_score, probs[0]
    except Exception as e:
        logger.error(f"Error in predict_image: {str(e)}")
        return None, None, None

# Report Generation Function (aligned with test.py)
def generate_report(patient_details, image, model_path):
    try:
        # Predict with ViT
        predicted_label, conf_score, probs = predict_image(image, model_path)
        if predicted_label is None:
            return "Error: Failed to process image or load model.", image

        # Generate patient details for display
        details = (
            f"Patient Details:\n"
            f"Name: {patient_details['name']}\n"
            f"Age: {patient_details['age']}\n"
            f"Gender: {patient_details['gender']}\n"
            f"Medical History: {patient_details['medical_history']}\n\n"
            f"Diagnosis:\n"
            f"Predicted Condition: {predicted_label}\n"
            f"Confidence Score: {conf_score:.2%}\n"
            f"Probabilities for all classes:\n"
        )
        for cls, prob in zip(CLASSES, probs):
            details += f"{cls}: {prob.item():.2%}\n"
        logger.info(f"Prediction: {predicted_label}, Confidence: {conf_score:.2%}")

        # Generate the detailed report using the Grok model
        report = MedicalReportChain.invoke({
            "name": patient_details['name'],
            "age": patient_details['age'],
            "gender": patient_details['gender'],
            "medical_history": patient_details['medical_history'],
            "condition": predicted_label,
            "confidence_score": conf_score
        })

        # Combine the patient details with the generated report
        final_report = details + "\nMedical Report:\n" + report
        return final_report, image
    except Exception as e:
        error_msg = f"Error generating report: {str(e)}"
        logger.error(error_msg)
        return error_msg, image

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Flask Chatbot for Diabetic Retinopathy Diagnosis")
parser.add_argument("--model_path", type=str, default="model_round_15.pth",
                   help="Path to the model weights")
args = parser.parse_args()

# Flask Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    report = None
    form_data = {"name": "", "age": "", "gender": "", "medical_history": ""}
    image_path = None

    if request.method == 'POST':
        # Get form data
        name = request.form['name']
        age = request.form['age']
        gender = request.form['gender']
        medical_history = request.form['medical_history']

        logger.info(f"Form Data: name={name}, age={age}, gender={gender}, medical_history={medical_history}")

        # Update form_data
        form_data = {
            "name": name,
            "age": age,
            "gender": gender,
            "medical_history": medical_history
        }

        if not all([name, age, gender, medical_history]) or 'image' not in request.files:
            report = "Error: All fields and an image are required."
        else:
            image_file = request.files['image']
            if image_file:
                try:
                    # Load and process image
                    image = Image.open(image_file).convert("RGB")

                    # Save the uploaded image
                    image_filename = "uploaded_image.jpg"
                    image_save_path = os.path.join(UPLOAD_FOLDER, image_filename)
                    image.save(image_save_path)
                    image_path = os.path.join('uploads', image_filename)

                    patient_details = {
                        "name": name,
                        "age": age,
                        "gender": gender,
                        "medical_history": medical_history
                    }
                    # Generate report
                    report, _ = generate_report(patient_details, image, args.model_path)
                except Exception as e:
                    report = f"Error processing image: {str(e)}"
                    logger.error(report)
            else:
                report = "Error: No image file uploaded."

    return render_template('index.html', report=report, form_data=form_data, image_path=image_path)

# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5003)