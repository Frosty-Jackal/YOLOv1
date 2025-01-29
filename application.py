from flask import Flask,url_for,redirect, send_from_directory,render_template,request
import os

from werkzeug.utils import secure_filename
app = Flask(__name__)
@app.route('/')
def form():
    return render_template('form.html')

@app.route('/submit', methods=['POST'])
def submit():
    file = request.files['img']
    filename = secure_filename(file.filename)
    filepath = os.path.join('uploads', filename)
    file.save(filepath)
    return redirect(url_for('show_image', filename=filename))

@app.route('/uploads/<filename>')
def show_image(filename):
    return send_from_directory('uploads', filename)


if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)