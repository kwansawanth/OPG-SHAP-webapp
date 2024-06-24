from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'

@app.route('/')
def index():
    return render_template('testpage.html')

@app.route('/frompredict-page1', methods=['POST'])
def predict_box():
    global model_select_input, predict_input_page1, node0_input, node1_input
    model_select_input = request.form.get('model_select')
    predict_input_page1 = request.form.get('frompredict')
    node0_input = request.form.get('node0input')
    node1_input = request.form.get('node1input')

    print(f'model_select_input: {model_select_input}')
    print(f'predict_input_page1: {predict_input_page1}')
    print(f'node0_input: {node0_input}')
    print(f'node1_input: {node1_input}')
    return render_template('shappage.html', predict_input=predict_input_page1, node0_input=node0_input, node1_input=node1_input)

@app.route('/frompredict', methods=['POST'])
def predict():
    global predict_input_page1, node0_input, node1_input
    predict_input = request.form.get('frompredict')
    node0_input = request.form.get('node0input')
    node1_input = request.form.get('node1input')
    return redirect(url_for('shappercentile', predict=predict_input, node0=node0_input, node1=node1_input))

@app.route('/shappercentile')
def evaluationpage():
    predict_input = request.args.get('predict')
    node0_input = request.args.get('node0')
    node1_input = request.args.get('node1')
    return render_template('shappercentile.html', predict=predict_input, node0=node0_input, node1=node1_input)

if __name__ == '__main__':
    app.run(debug=True)



####
