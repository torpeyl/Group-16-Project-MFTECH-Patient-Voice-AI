from .overall_eval import overall_eval

def eval(data):
    
    return overall_eval(sentiment_score=float(data["result_sentiment_model"]),
                                                 SDR=float(data["SDR"]),
                                                 disease_index=data["disease_index"],
                                                 disease_prob=float(data["disease_prob"]),
                                                 )
    
