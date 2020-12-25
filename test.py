from flask import Flask,request,render_template, redirect, url_for
import pickle
import pandas as pd
from joblib import dump,load
from sklearn.preprocessing import LabelEncoder
app = Flask(__name__)
classifier = load('file.joblib')



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/class1')
def class1():
    return render_template('class1.html')

@app.route('/class2')
def class2():
    return render_template('class2.html')

@app.route('/class3')
def class3():
    return render_template('class3.html')

@app.route('/class4')
def class4():
    return render_template('class4.html')

@app.route('/class5')
def class5():
    return render_template('class5.html')

@app.route('/class6')
def class6():
    return render_template('class6.html')

@app.route('/class7')
def class7():
    return render_template('class7.html')

@app.route('/class8')
def class8():
    return render_template('class8.html')

@app.route('/class9')
def class9():
    return render_template('class9.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/classes')
def classes():
    return render_template('classes.html')




@app.route('/',methods=['POST','GET'])
def savepost():
    if request.method=='POST':
        a=request.form['u']
        b=request.form['a']
        data = [[a, b]]
        n1 = pd.DataFrame(data)
        n2 = n1.select_dtypes(include=["object"])
        n3 = n1.select_dtypes(include=["integer", "float"])
        label = LabelEncoder()
        n4 = n2.apply(label.fit_transform)
        n4 = n4.join(n3)
        n4.head()
        new_input = [n4]
        new_output = classifier.predict(new_input)
        print(new_output)
        return render_template('index.html', n=new_output)
    else:
        return "error"

if __name__ == "__main__":
    app.run(debug=True)