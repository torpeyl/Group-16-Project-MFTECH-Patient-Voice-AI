from flask import Flask, request, make_response, render_template, Response, send_from_directory, jsonify
import os
import sys
import subprocess
from Illness_Determination_Model.main import illness_classifier as classifier

app = Flask(__name__, template_folder='Templates')

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/convert_wav', methods=['POST'])
def convert_wav():
    file = request.files['file']

    if not os.path.exists('downloads'):
        os.makedirs('downloads')

    filename = 'recording.wav'
    file.save(f"./downloads/{filename}")

    result = classifier(f'/Users/Dev/Documents/Back End/VAI_Back_End/Modules/Flask_App/downloads/{filename}')
    print(result)
    return render_template('download.html', result=result)
     
@app.route('/download/<result>')
def download(result):
    #return send_from_directory('downloads', filename, download_name='result.wav', as_attachment=True)
    return f"Result: {result}"



if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5001', debug=True)