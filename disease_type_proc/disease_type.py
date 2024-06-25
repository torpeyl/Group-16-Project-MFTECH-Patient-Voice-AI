import json

# Load the JSON file from local storage
with open('./disease_type/mapping.json', 'r') as file:
    data = json.load(file)

with open('./disease_type/diseases.txt', 'w') as textfile:
    for disease,label in data.items():
        textfile.write(f"{label}. {disease}\n")
        
print("finished")

