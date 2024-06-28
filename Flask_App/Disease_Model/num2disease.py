from .disease_mapping import find_disease

def num2disease(model_output):

    disease = find_disease(model_output)

    return disease