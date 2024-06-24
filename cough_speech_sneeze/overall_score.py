def overall_eval(
        sentiment_score:float, 
        disruption_ratio:float, 
        disease:str, 
        disease_prob:float, 
        consult_threshold:float=0.5, 
        w_senti:float=0.5,      # essentially an OR gate
        treatment_threshold:float=0.5, 
        mapping:dict=None
        ):
    # scale the sentiment score onto the range [0,1]
    sentiment_score = sentiment_score*0.5 + 0.5

    # naive classifier
    # 1. further consultation
    consult = bool(disease_prob < consult_threshold)

    # 2. urgency for treatment
    treatment_score = sentiment_score*w_senti + disruption_ratio*(1-w_senti)
    treatment = bool(treatment_score > treatment_threshold)

    return consult,treatment

test_result = overall_eval(-0.8,0.3,'cold',0.91)
print(test_result)