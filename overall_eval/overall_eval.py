import json

# Load the JSON files from local storage
with open('./overall_eval/disruptions.json', 'r') as file1:
    disruptions_type = json.load(file1)
with open('./overall_eval/emotions.json', 'r') as file2:
    emotions_type = json.load(file2)

def overall_eval(
        sentiment_score:float, 
        disruption_ratio:float, 
        disease:str, 
        disease_prob:float, 
        consult_threshold:float=0.5, 
        treatment_threshold:float=0.5
        ):
    
    # sentiment score input: [-1,1] => [max_positive, max_negative]
    # scale onto the range [0,1] => sentiment: [max_positive (healthy), max_negative (sick)]
    sentiment_score = (sentiment_score)*0.5 + 0.5
    # disruption ratio input: [0,1] => [no_disruption (healthy), all_disruption (sick)]

    # 1. further consultation (yes or no)
    consult = bool(disease_prob < consult_threshold)

    # 2. urgency for treatment
    dis_flag = disease in disruptions_type
    emo_flag = disease in emotions_type
    # determine the weightings based on disease type
    if not (dis_flag or emo_flag):
        w_senti = 0.3
        w_dis = 0.3
    else:
        if (dis_flag and emo_flag):     # both factors are important
            w_senti = 0.5   # average
        elif dis_flag:      # disruption ratio is more important
            w_senti = 0.3
        elif emo_flag:      # sentiment score is more important
            w_senti = 0.7
        w_dis = 1-w_senti
    # calculate treatment score (higher means more ill)
    treatment_score = round(sentiment_score*w_senti + disruption_ratio*w_dis,3)
    print("treatment score:", treatment_score)
    # convert to yes or no
    treatment = bool(treatment_score > treatment_threshold)

    return consult,treatment


# example:
test_result = overall_eval(sentiment_score=0.5,disruption_ratio=0.7,disease="Asthma",disease_prob=0.91)
print(test_result)

