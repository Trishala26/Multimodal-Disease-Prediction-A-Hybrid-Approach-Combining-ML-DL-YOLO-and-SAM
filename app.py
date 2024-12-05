from flask import Flask, render_template, request, send_from_directory
import numpy as np
import pandas as pd
import pickle
import joblib
import os
import cv2
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)

# Load the trained disease prediction models
model_path_heart = r"C:\Users\trilo\OneDrive\Desktop\Trishala\Final-Year-project\PickleFiles\heart (1).pkl"
heart = joblib.load(model_path_heart)
model_path_diabetes = r"C:\Users\trilo\OneDrive\Desktop\Trishala\Final-Year-project\PickleFiles\diabetes (1).pkl"
diabetes_model = joblib.load(model_path_diabetes)  # Renaming the model variable
model_path_liver = r"C:\Users\trilo\OneDrive\Desktop\Trishala\Final-Year-project\PickleFiles\liver_yes_no (1).pkl"
liver = joblib.load(model_path_liver)
model_path_thyroid = r"C:\Users\trilo\OneDrive\Desktop\Trishala\Final-Year-project\PickleFiles\Thyroid (1).pkl"
thyroid = joblib.load(model_path_thyroid)
model_path_stroke = r"C:\Users\trilo\OneDrive\Desktop\Trishala\Final-Year-project\PickleFiles\stroke.pkl"
stroke = joblib.load(model_path_stroke)
model_path_copd = r"C:\Users\trilo\OneDrive\Desktop\Trishala\Final-Year-project\PickleFiles\copd.pkl"
copd = joblib.load(model_path_copd)

# YOLO model setup
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DETECTED_FOLDER'] = 'detected'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DETECTED_FOLDER'], exist_ok=True)

# Load the YOLO model explicitly specifying CPU
model = YOLO(r"C:\Users\trilo\Downloads\Final-year-project1\static\best.pt").to('cpu')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/index.html')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/chat.html')
def chatbot():
    return render_template('chatbot.html')

@app.route('/diseases')
def diseases():
    return render_template('diseases.html')

@app.route('/brain_tumor')
def brain_tumor():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file'
    
    # Save the uploaded file to the uploads folder
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Load the image
    img = Image.open(file_path)
    
    # Perform inference
    results = model(img)

    # Save the detected image with bounding boxes
    detected_image_path = os.path.join(app.config['DETECTED_FOLDER'], file.filename)
    
    # Convert the results to an image (numpy array) and save it
    result_image = results[0].plot()  # Get the numpy array with detection
    cv2.imwrite(detected_image_path, result_image)  # Save using OpenCV

    # Check for brain tumor detection
    has_tumor = False  # Initialize the variable to track tumor presence

    # Iterate through the results
    for result in results:
        for box in result.boxes:
            if box.cls == 0:  # Assuming 'tumor' is class ID 0, adjust if necessary
                has_tumor = True
                break  # Exit the loop if a tumor is found
        if has_tumor:
            break  # Exit the outer loop if a tumor is found

    # Remove the uploaded file after processing
    os.remove(file_path)

    # Pass the detected image path to the template
    message = 'The image shows signs of a brain tumor.' if has_tumor else 'The image shows signs of a brain tumor.'
    return render_template('upload.html', message=message, detected_image=file.filename)

@app.route('/detected/<filename>')
def send_detected_image(filename):
    return send_from_directory(app.config['DETECTED_FOLDER'], filename)

# Disease prediction routes

@app.route('/heart_disease')
def heart_disease():
    return render_template('heart.html')

@app.route('/make_prediction', methods=['POST'])
def make_prediction_heart():
    if request.method == 'POST':
        features = [float(x) for x in request.form.values()]
        input_data = np.array(features).reshape(1, -1)
        prediction = heart.predict(input_data)
        result = "HEART DISEASE" if prediction[0] == 1 else "NO HEART DISEASE"
        return render_template('prediction_heart_result.html', result=result)

@app.route('/diabetes')
def diabetes():  # Renaming the function to avoid conflict
    return render_template('diabetes.html')

@app.route('/make_prediction1', methods=['POST'])
def make_prediction_diabetes():
    if request.method == 'POST':
        features = [float(x) for x in request.form.values()]
        input_data = np.array(features).reshape(1, -1)
        prediction = diabetes_model.predict(input_data)  # Update to use the renamed model variable
        result = "Diabetes" if prediction[0] == 1 else "Healthy"
        return render_template('prediction_diabetes.html', result=result)


@app.route('/liver_disease')
def liver_disease():
    return render_template('liver.html')

@app.route('/make_prediction2', methods=['POST'])
def make_prediction_liver():
    if request.method == 'POST':
        features = [float(x) for x in request.form.values()]
        input_data = np.array(features).reshape(1, -1)
        prediction = liver.predict(input_data)
        result = "Has Some Liver Disease" if prediction[0] == 1 else "Healthy"
        return render_template('prediction_liver.html', result=result)

@app.route('/thyroid_disease')
def thyroid_disease():
    return render_template('Thyroid.html')

@app.route('/make_prediction4', methods=['POST'])
def make_prediction_thyroid():
    if request.method == 'POST':
        features = [float(x) for x in request.form.values()]
        input_data = np.array(features).reshape(1, -1)
        prediction = thyroid.predict(input_data)
        result = "Has parathyroid" if prediction[0] == 1 else "Healthy"
        return render_template('prediction_thyroid.html', result=result)

@app.route('/stroke_disease')
def stroke_disease():
    return render_template('stroke.html')

@app.route('/make_prediction5', methods=['POST'])
def make_prediction_stroke():
    if request.method == 'POST':
        features = [float(x) for x in request.form.values()]
        input_data = np.array(features).reshape(1, -1)
        prediction = stroke.predict(input_data)
        result = "Has Stroke" if prediction[0] == 1 else "Healthy"
        return render_template('prediction_stroke.html', result=result)

@app.route('/copd_disease')
def copd_disease():
    return render_template('Copd.html')

@app.route('/make_prediction6', methods=['POST'])
def make_prediction_copd():
    if request.method == 'POST':
        features = [float(x) for x in request.form.values()]
        input_data = np.array(features).reshape(1, -1)
        prediction = copd.predict(input_data)
        if prediction[0] == 'MILD':
            result = 'The Patient has mild COPD'
        elif prediction[0] == 'MODERATE':
            result = 'The Patient has moderate COPD'
        elif prediction[0] == 'SEVERE':
            result = 'The Patient has severe COPD'
        elif prediction[0] == 'VERY SEVERE':
            result = 'The Patient has very severe COPD'
        else:
            result = 'The Patient is Healthy'
        return render_template('prediction_copd.html', result=result)

# Drug Recommendation
sym_des = pd.read_csv(r"C:\Users\trilo\OneDrive\Desktop\Trishala\Final-Year-project\Medicine-Recommendation-System\dataset\symtoms_df.csv")
precautions = pd.read_csv(r"C:\Users\trilo\OneDrive\Desktop\Trishala\Final-Year-project\Medicine-Recommendation-System\dataset\precautions_df.csv")
workout = pd.read_csv(r"C:\Users\trilo\OneDrive\Desktop\Trishala\Final-Year-project\Medicine-Recommendation-System\dataset\workout_df.csv")
description = pd.read_csv(r"C:\Users\trilo\OneDrive\Desktop\Trishala\Final-Year-project\Medicine-Recommendation-System\dataset\description.csv")
medications = pd.read_csv(r"C:\Users\trilo\OneDrive\Desktop\Trishala\Final-Year-project\Medicine-Recommendation-System\dataset\medications.csv")
diets = pd.read_csv(r"C:\Users\trilo\OneDrive\Desktop\Trishala\Final-Year-project\Medicine-Recommendation-System\dataset\diets.csv")

svc = pickle.load(open(r"C:\Users\trilo\OneDrive\Desktop\Trishala\Final-Year-project\Medicine-Recommendation-System\models\svc.pkl",'rb'))


def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]

    med = medications[medications['Disease'] == dis]['Medication']
    med = [med for med in med.values]

    die = diets[diets['Disease'] == dis]['Diet']
    die = [die for die in die.values]

    wrkout = workout[workout['disease'] == dis] ['workout']


    return desc,pre,med,die,wrkout

symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    return diseases_list[svc.predict([input_vector])[0]]



@app.route("/home.html")
def index1():
    return render_template("home.html")

@app.route('/predict', methods=['GET', 'POST'])
def home1():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms')
        # mysysms = request.form.get('mysysms')
        # print(mysysms)
        print(symptoms)
        if symptoms =="Symptoms":
            message = "Please either write symptoms or you have written misspelled symptoms"
            return render_template('home.html', message=message)
        else:

            user_symptoms = [s.strip() for s in symptoms.split(',')]
            user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms]
            predicted_disease = get_predicted_value(user_symptoms)
            dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease)

            my_precautions = []
            for i in precautions[0]:
                my_precautions.append(i)

            return render_template('home.html', predicted_disease=predicted_disease, dis_des=dis_des,
                                   my_precautions=my_precautions, medications=medications, my_diet=rec_diet,
                                   workout=workout)

    return render_template('home.html')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)