from .speech_to_text import speech2text
from .classifier_pipeline import disease_classifier as classifier
from .num2disease import num2disease

def disease_model(file_path):

    return classifier(speech2text(file_path))