from flask import Flask,render_template,request,redirect,url_for,abort
import numpy as np
import pandas as pd 
import pickle 


app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('abalone.html')



@app.route('/prediction',methods = ['GET', 'POST'])
def pred():
    if request.method == 'POST': 
        
       q = request.form['sex'] 
       data = dict(request.form)
       print(dict(request.form))   
       print(request.form.values())
       features = [float(x[1]) for x in data.items() if x[0] != 'sex'   ]
       print("========",features)
       print(data.items())
       
       
       
        
            
       if q == 'M': 
          c,e,r = 0,0,1
       elif q== 'F':
          c,e,r = 1,0,0
       elif q =='I':
          c,e,r = 0,1,0
       s = [c,e,r]    
       features.extend(s)   
       print(features)
       out =  model.predict([features])   
        
       
       return render_template('abalone.html', output = f"The predicted age is: {out+1.5}" )
       


              
if __name__ == '__main__':
    app.run(debug = True)
   