import pickle
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the trained model
with open('models\X_pca.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input features from the user
        General_Health = int(request.form['General_Health'])
        Checkup = int(request.form['Checkup'])
        Exercise= int(request.form['Exercise'])
        Skin_Cancer = int(request.form['Skin_Cancer'])
        Other_Cancer = int(request.form['Other_Cancer'])
        Depression = int(request.form['Depression'])
        Arthritis = int(request.form['Arthritis'])
        Smoking_History = int(request.form['Smoking_History'])
        Diabetes = int(request.form['Diabetes'])
        Age_Category = float(request.form['Age_Category'])
        Height = float(request.form['Height_(cm)'])
        Weight = float(request.form['Weight_(kg)'])
        BMI = float(request.form['BMI'])
        Alcohol_Consumption = float(request.form['Alcohol_Consumption'])
        Fruit_Consumption = float(request.form['Fruit_Consumption'])
        Green_Vegetables_Consumption = float(request.form['Green_Vegetables_Consumption'])
        
        

        # Make the prediction
        prediction = model.predict([[General_Health,Checkup,Exercise,Skin_Cancer,Other_Cancer,Depression,Arthritis,Smoking_History,Diabetes,Age_Category,Height,Weight,BMI,Alcohol_Consumption,Fruit_Consumption,Green_Vegetables_Consumption]])
        result = "Positive" if prediction[0] == 1 else "Negative"

        return render_template('index.html', prediction_result=result)
    except Exception as e:
        return render_template('index.html', prediction_result="Error: " + str(e))

if __name__ == '__main__':
    app.run(debug=True)
