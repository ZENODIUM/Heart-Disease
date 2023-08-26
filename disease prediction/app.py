from flask import Flask,render_template,request
import pickle
import tensorflow
import numpy as np
import keras
import joblib
import sklearn.neighbors._base
import numpy as np
from tensorflow import keras
import cv2
from keras.models import load_model
app = Flask(__name__)
@app.route('/',methods=['GET'])
def hello_world():
    return render_template("index.html")

@app.route('/heart',methods=['GET', 'POST'])
def heart():
    if request.method=='POST':
        with open('saved_model','rb') as f:
            gender = request.form.get('gender')
            age = request.form.get('age')
            pain_type = request.form.get('pain_type')
            bp = request.form.get('bp')
            cholesterol = request.form.get('cholesterol')
            sugar = request.form.get('sugar')
            restecg = request.form.get('restecg')
            heart_rate = request.form.get('heart_rate')
            exang = request.form.get('exang')
            oldpeak = request.form.get('oldpeak')
            slope = request.form.get('slope')
            ca = request.form.get('ca')
            thal = request.form.get('thal')
            input_data = (age,gender,pain_type,bp,cholesterol,sugar,restecg,heart_rate,exang,oldpeak,slope,ca,thal)

            print(gender)
            print(age)
            print(pain_type)
            print(bp)
            print(cholesterol)
            print(sugar)
            print(restecg)
            print(heart_rate)
            print(exang)
            print(oldpeak)
            print(slope)
            print(ca)
            print(thal)
            
            input_npar = np.asarray(input_data)
            
            input_npar = np.array(input_npar, dtype=float)
            input_reshaped = input_npar.reshape(1,-1)

            
            ann = load_model("annn.h5")
            svm_pickle = pickle.load(open('svm_pickle','rb'))
            xgb_pickle = pickle.load(open('xgb_pickle','rb'))
            
            rf_pickle = pickle.load(open('rf_pickle','rb'))
            loaded_model = pickle.load(open('saved_model','rb'))
            tree_pickle = pickle.load(open('tree_pickle','rb'))
            prediction = loaded_model.predict(input_reshaped)
            ann_result = ann.predict(input_reshaped)
            prediction_svm = svm_pickle.predict(input_reshaped)
            prediction_xgb = xgb_pickle.predict(input_reshaped)
            prediction_rf = rf_pickle.predict(input_reshaped)
            
            prediction_tree = tree_pickle.predict(input_reshaped)
            disease = 0
            no_disease = 0
            if (prediction[0] == 1):
                result = "Logistic regression: Heart Disease"
                disease += 1
            else:
                result = "Logistic regression: Probable no Heart Disease"
                no_disease += 1
            if (prediction_svm[0] == 1):
                result_svm = "SVM: Heart Disease"
                disease += 1
            else:
                result_svm = "SVM: Probable no Heart Disease"
                no_disease += 1
            if (prediction_tree[0] == 1):
                result_tree = "Decision Tree: Heart Disease"
                disease += 1
            else:
                result_tree = "Decision Tree: Probable no Heart Disease"
                no_disease += 1

            if (prediction_rf[0] == 1):
                result_rf = "Random Forest: Heart Disease"
                disease += 1
            else:
                result_rf = "Random Forest: Probable no Heart Disease"
                no_disease += 1

            if (prediction_xgb[0] == 1):
                result_xgb = "XGB: Heart Disease"
                disease += 1
            else:
                result_xgb = "XGB: Probable no Heart Disease"
                no_disease += 1

            if (ann_result[0] == 1):
                result_ann = "ANN: Heart Disease"
                disease += 1
            else:
                result_ann = "ANN: Probable no Heart Disease"
                no_disease += 1

            if no_disease >= disease:
                result_ensemble = "Ensemble model: No Heart Disease"
            else:
                result_ensemble = "Ensemble model: Heart Disease"
            
            
            



        return render_template("result.html",result_ann = result_ann,result_ensemble= result_ensemble,result_xgb= result_xgb,result_rf = result_rf,result_tree = result_tree,result_svm=result_svm,result = result,bp = bp,max_chol = 220 - int(age),cholesterol = cholesterol,oldpeak=oldpeak,heart_rate=heart_rate)

    return render_template("heart.html")

    


if __name__ == '__main__':
    app.run(port=3000,debug=True)