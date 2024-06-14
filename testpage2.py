from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'


@app.route('/', methods=['GET', 'POST'])
def predict_box():
    return render_template('testpage.html')

@app.route('/frompredict', methods=['POST'])
def predict():
    predict_input = request.form.get('frompredict')
    node0_input = request.form.get('node0input')
    node1_input = request.form.get('node1input')

    return redirect(url_for('evaluationpage', predict=predict_input, node0=node0_input, node1=node1_input))

@app.route('/evaluationpage')
def evaluationpage():
    predict_input = request.args.get('predict')
    node0_input = request.args.get('node0')
    node1_input = request.args.get('node1')

    return render_template('evaluationpage.html', predict=predict_input, node0=node0_input, node1=node1_input)

if __name__ == '__main__':
    app.run(debug=True)
