from flask import Flask, request, make_response, render_template, Response, send_from_directory, jsonify, url_for
from flask_mail import Mail, Message
import os
from pydub import AudioSegment
import requests
#Model Imports

from Cough_Model.main import cough_model
from Disease_Model.main import disease_model
from Sentiment_Model.script import sentiment_model
from Evaluator.main import eval


from Data2Worddoc.WorddocGenerator import toDoc

app = Flask(__name__, template_folder='Templates', static_folder='Static', static_url_path='/')

# Configuration for Flask-Mail using environment variables
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USERNAME'] = 'mftechproject@gmail.com' # Mailtrap username
#app.config['MAIL_PASSWORD'] =   Mailtrap password
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
#app.config['MAIL_DEFAULT_SENDER'] = 'mftechproject@gmail.com'
mail = Mail(app)


@app.route('/', methods=['GET', 'POST'])
def index():

    return render_template('home.html')

data = {}

@app.route('/convert_wav', methods=['POST'])
def convert_wav():

    data.clear()

    data["name"] = request.form['name']
    data["age"] = request.form['age']
    data["pain_level"] = request.form['points']
    data["birth_gender"] = request.form['inlineRadioOptions']
    data["smoker"] = 'Y' if request.form.get('smoker') == 'Y' else 'N'
    data["drinker"] = 'Y' if request.form.get('drinker') == 'Y' else 'N'
    file = request.files['file']

    '''print(request.form.get("live_recording"))
    src = 'recording.ogg'
    # convert mp3 to wav
    sound = AudioSegment.from_ogg(src)
    sound.export('please.wav', format="wav")


    URL = request.form.get('audioURL')
    #response = requests.get(URL, stream=True)
    print(request.form.get('live_recording'))
    file = request.files['file']
    live_file = request.files.get('live_recording')

    print(request.files.get('live_recording'))
    
    upload_path = "./downloads/test.ogg"
    live_file.save(upload_path)

    sound = AudioSegment.from_ogg(upload_path)
    sound.export('please.wav', for mat="wav")'''


    if not os.path.exists('downloads'):
        os.makedirs('downloads')

    filename = 'recordings.wav'
    upload_path = f"./downloads/{filename}"
    file.save(upload_path)
    
    #Code for Anlan's Model
    result_disease_model, index, prob = disease_model(upload_path)
    data["result_disease_model"] = result_disease_model
    data["disease_index"] = index
    data["disease_prob"] = prob
    
    #Code for Daisy's Model
    result_cough_model, SDR = cough_model(upload_path)
    result_cough_model = result_cough_model[8:]
    data["result_cough_model"] = result_cough_model
    data["SDR"] = SDR

    #Code for Ed's Model
    result_sentiment_model = sentiment_model(upload_path)
    data["result_sentiment_model"] = result_sentiment_model

    #Evaluation
    consult_bool, treatment_bool, treatment_score = eval(data)
    data['consult_bool'] = consult_bool
    data['treatment_bool'] = treatment_bool
    data['treatment_score'] = treatment_score

    #Data --> .docx --> .pdf
    toDoc(data)

    return render_template('results.html', data=data)


@app.route('/handle_email', methods=['POST'])
def handle_email():
    client_email = request.json['email']

    #Sends PDF as email
    msg = Message(subject='MFTech Diagnosis Results',
                sender='mftechproject@gmail.com',
                recipients=[f'{client_email}'])
    msg.body = f"Hey {data['name']}, here are your AI diagnosis results!"

    with app.open_resource("Data2Worddoc/diagnosis.pdf") as fp:
        msg.attach(f"diagnosis_{data['name']}.pdf", "application/pdf", fp.read())

    with app.open_resource("downloads/uploaded_file.png") as fp:
        msg.attach('speech_disruption_plot.png', content_type='image/png', data=fp.read())

    mail.send(msg)

    return jsonify({'Message': "The results have been sent to your email!"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5001', debug=True)