@app.route('/', methods=['POST'])
def predict():
    global predict_input, node0_input, node1_input
    predict_input = request.form.get('frompredict')
    node0_input = request.form.get('node0input')
    node1_input = request.form.get('node1input')