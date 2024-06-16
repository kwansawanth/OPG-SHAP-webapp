from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/frompredict-page1', methods=['GET', 'POST'])
def predict_box():
    global predict_input_page1
    predict_input_page1 = request.form.get('frompredict')

    return render_template('testpage.html', predict_input=predict_input_page1)

@app.route('/frompredict', methods=['POST'])
def predict():
    global predict_input, node0_input, node1_input
    predict_input = predict_input_page1
    node0_input = request.form.get('node0input')
    node1_input = request.form.get('node1input')

    return redirect(url_for('evaluationpage', predict=predict_input, node0=node0_input, node1=node1_input))

@app.route('/evaluationpage')
def evaluationpage():
    global predict_input, node0_input, node1_input
    return render_template('evaluationpage.html', predict=predict_input, node0=node0_input, node1=node1_input)

if __name__ == '__main__':
    app.run(debug=True)
