from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'

# กำหนดตัวแปร global สำหรับเก็บค่าที่ถูกส่งมาจากฟอร์ม
predict_input = None
node0_input = None
node1_input = None

@app.route('/', methods=['GET', 'POST'])
def predict_box():
    return render_template('shappage.html')

@app.route('/frompredict', methods=['POST'])
def predict():
    global predict_input, node0_input, node1_input
    predict_input = request.form.get('frompredict')
    node0_input = request.form.get('node0input')
    node1_input = request.form.get('node1input')
    
    # ทำการ redirect ไปยัง evaluationpage พร้อมส่งค่าที่ได้รับจากฟอร์มไปด้วย
    return redirect(url_for('evaluationpage', predict=predict_input, node0=node0_input, node1=node1_input))

@app.route('/evaluationpage')
def evaluationpage():
    global predict_input, node03_input, node1_input
    # ส่งค่าที่ได้รับจากฟอร์มไปยัง evaluationpage
    return render_template('evaluationpage.html', predict=predict_input, node0=node0_input, node1=node1_input)

if __name__ == '__main__':
    app.run(debug=True)

#if เขาเลือกโมเดลเรา set id name รับค่าจากflask set default ว่าโมเดลนี้เป็นอะไร
