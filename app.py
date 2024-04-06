# here we doing web application where we having form , 
# here we give all our input data that is required to presict students performances 
# here we considering using flask app 


from flask import Flask , request , render_template  
import numpy as np 
import pandas as pd 

from sklearn.preprocessing import StandardScaler   
from src.pipeline.predict_pipeline import CustomData ,predictpipeline 

application= Flask(__name__) 
app=application 

@app.route('/') 
def index():
    return render_template('index.html')


@app.route('/predict' , methods=['GET','POST'])  

def predict_datapoint():
    if request.method == 'GET' :
        return render_template('form.html') 
    
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            age=float(request.form.get('age')),
            hypertension=int(request.form.get('hypertension')),
            heart_disease=int(request.form.get('heart_disease')),
            smoking_history=request.form.get('smoking_history'),
            bmi=float(request.form.get('bmi')),
            HbA1c_level=float(request.form.get('HbA1c_level')),
            blood_glucose_level=int(request.form.get('blood_glucose_level'))
            
        )
        
        
        final_new_data=data.get_data_as_data_frame()
        predict_pipeline=predictpipeline()
        pred=predict_pipeline.predict(final_new_data)
        
        result_str = 'Positive' if pred[0] == 1 else 'Negative'
        
        # Determine message based on prediction
        message = "The prediction result is Positive.You are Diabetic " if pred[0] == 1 else "The prediction result is Negative. You are Non Diabetic"
        
        # Render results template with prediction
        return render_template('results.html', final_result=result_str, message=message)
    
    
    
    
if __name__ == '__main__': 
    app.run(host='0.0.0.0',debug=True,port=5000) 