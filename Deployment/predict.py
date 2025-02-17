#LOAD MODEL
import pickle
from flask import Flask
from flask import request #converts reuqest into json
from flask import jsonify  #converts response into json
from waitress import serve

#import model
model_file = 'model_C=1.0.bin'

#load model
with open(model_file, 'rb') as f_in: #rb=read bin, 
    dv, model = pickle.load(f_in)

#get a random customer
customer = {
    'gender': 'female',
    'seniorcitizen': 0,
    'partner': 'yes',
    'dependents': 'no',
    'phoneservice': 'no',
    'multiplelines': 'no_phone_service',
    'internetservice': 'dsl',
    'onlinesecurity': 'no',
    'onlinebackup': 'yes',
    'deviceprotection': 'no',
    'techsupport': 'no',
    'streamingtv': 'no',
    'streamingmovies': 'no',
    'contract': 'month-to-month',
    'paperlessbilling': 'yes',
    'paymentmethod': 'electronic_check',
    'tenure': 1,
    'monthlycharges': 29.85,
    'totalcharges': 29.85
}

app= Flask('churn')
@app.route('/predict', methods = ['POST'])

def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0,1]
    churn = y_pred >= 0.5
    
    results =  {
        'churn_probability' : float(y_pred),
        'churn': bool(churn)
    }
    return jsonify(results)

if __name__ == "__main__":
    serve(app, host='0.0.0.0', port=8080)

#waitress is a WSGI server used for production deployment.
    #from waitress import serve
    #serve(app, host="0.0.0.0", port=8080)

    ## Dcoker commands
    ##build - docker build -t name of build(zoomcamp-test . )
    ##run - docker run -it --rm --entrypoint:bash -p 8080:8080 --zoomcamp-test
    ##waitress cli - waitress-serve --listen=0.0.0.0:port predict:app