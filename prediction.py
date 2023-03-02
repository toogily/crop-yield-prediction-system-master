from flask import Flask,render_template,request,redirect
from flask_cors import CORS,cross_origin
import pickle
import pandas as pd
import numpy as np

import sklearn
print(sklearn.__version__)
import sys
print(sys.executable)

app = Flask(__name__)
cors=CORS(app)
model=pickle.load(open('LinearRegressionModel.pkl','rb'))
crop=pd.read_csv('crops-data.csv')


@app.route('/', methods=['GET','POST'])
def index():

    counties = sorted(crop['county'].unique())
    sub_regions = sorted(crop['sub_name'].unique())
    items = sorted(crop['item'].unique())
    seasons = sorted(crop['season'].unique())


    counties.insert(0,'Select county')
    items.insert(0, 'Select crop')
    seasons.insert(0, 'Select county')
    return render_template('index.html', counties=counties,sub_regions=sub_regions,items=items,seasons=seasons)


@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():

    county = request.form.get('county')
    sub_region = request.form.get('sub_regions')
    item = request.form.get('item')
    season = request.form.get('season')


    prediction = model.predict(pd.DataFrame(columns=['county','sub_name','item','season'], data=np.array([county,sub_region,item,season]).reshape(1,4)))

    print(prediction)

    return str(np.round(prediction[0], 5))


if __name__ == '__main__':
    app.run()