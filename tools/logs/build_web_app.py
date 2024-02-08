# 將 training curve 顯示在網頁上
from flask import Flask, render_template, url_for
app = Flask(__name__)

@app.route('/')
def training_curve():
    image_url = url_for('static', filename='test.png')
    return render_template('training_curve.html', image_url=image_url)

if __name__ == '__main__':
    app.run(debug=True)