from docxtpl import DocxTemplate, InlineImage
from docx.shared import Mm
from datetime import datetime
from docx2pdf import convert
from Disease_Model.speech_to_text import speech2text

def toDoc(data):
    doc = DocxTemplate("./Data2Worddoc/diagnosis_template.docx")
    date = datetime.today().strftime("%d/%m/%y")
    name = data["name"]
    age = data["age"]
    pain = data["pain_level"]
    smoker = data["smoker"]
    drinker = data["drinker"]

    transcript = speech2text("./downloads/recordings.wav")

    image_path = InlineImage(doc, image_descriptor=f"Static/{data['result_cough_model']}", width=Mm(175), height=Mm(20))
    SDR = data["SDR"]
    disease_one = data["result_disease_model"][0]
    disease_two = data["result_disease_model"][1]
    disease_three =  data["result_disease_model"][2]
    sentiment_ratio = data["result_sentiment_model"]

    if data["consult_bool"] == "True":
        consult_bool = "Y"
    else:
        consult_bool = "N"

    if data["treatment_bool"] == "True":
        treatment_bool = "Y"
    else:
        treatment_bool = "N"

    treatment_score = data["treatment_score"]

    context = {
        "date": date,
        "name": name,
        "age": age,
        "pain": pain,
        "smoker": smoker,
        "drinker": drinker,
        "transcript": transcript,
        "image_path": image_path,
        "SDR": SDR,
        "disease_one": disease_one,
        "disease_two": disease_two,
        "disease_three": disease_three,
        "sentiment_ratio": sentiment_ratio,
        "consult_bool": consult_bool,
        "treatment_bool": treatment_bool,
        "treatment_score": treatment_score

    }

    doc.render(context)
    save_path = "Static/files/diagnosis.docx"
    doc.save(save_path)
    convert(save_path)
    pdf_save_path = "Static/files/diagnosis.pdf"
    return save_path

