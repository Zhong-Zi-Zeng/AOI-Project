# 將 curve 顯示在網頁上
from flask import Flask, render_template, url_for, request
import sys

app = Flask(__name__)

@app.route('/')
def index():
    # 只能先寫好
    loss_image_url = url_for('static', filename="Test_acc.png")
    accuracy_image_url = url_for('static', filename="Accuracy.png")
    return render_template('curve_analysis.html', loss_image_url=loss_image_url, accuracy_image_url=accuracy_image_url)

if __name__ == '__main__':
    app.run(debug=True)