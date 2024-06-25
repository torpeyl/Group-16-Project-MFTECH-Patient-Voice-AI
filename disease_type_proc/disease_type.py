import json

# Load the JSON file from local storage
with open('./disease_type_proc/mapping.json', 'r') as file:
    data = json.load(file)

# with open('./disease_type/diseases.txt', 'w') as textfile:
#     for disease,label in data.items():
#         textfile.write(f"{label}. {disease}\n")

with open('./disease_type_proc/extremes.json', 'r') as file1:
    extremes = json.load(file1)

for disease,index in extremes.items():
    correct_index = data[disease]
    extremes.update({disease:correct_index})


# Convert and write JSON object to file
with open("./disease_type_proc/extremes_1.json", "w") as outfile: 
    json.dump(extremes, outfile)
        
print("finished")

