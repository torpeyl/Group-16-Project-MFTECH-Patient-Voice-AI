import os
import librosa
from model import RNNSpeechModel
from model import calculate_health_indicators
from plotter import TimescalePlotter

data_path = "cough_speech_sneeze\\data"
results_path = "cough_speech_sneeze\\results"

# an example recording
filename = "qPo0bymnsh0.wav"
filepath = os.path.join(data_path, filename)
audio, sample_rate = librosa.load(filepath)

# create an model
model = RNNSpeechModel()
# predict for the example recording
y_labels,y_probs,y_speech = model.predict_framewise(audio)
# plot the results
plotter = TimescalePlotter(audio,sample_rate,y_labels,y_probs,y_speech,filename)
plotter.plot()


# do the same for all samples in the "data" folder
# and save the txt results in the "results" folder
model = RNNSpeechModel()
for filename in os.listdir(data_path):
    print(filename+": ")
    # load the audio
    f = os.path.join(data_path, filename)
    audio, sample_rate = librosa.load(f)

    # perform classification
    y_labels, y_probs, y_speech = model.predict_framewise(audio)

    # plot the classification results and timestamps
    plotter = TimescalePlotter(audio,sample_rate,y_labels,y_probs,y_speech,filename)
    plotter.plot()
    
    # calculate label ratios as health indicators
    speech_ratio, disruption_ratio = calculate_health_indicators(y_labels)
    # print("speech ratio: ", speech_ratio)
    # print("disruption ratio: ", disruption_ratio)

    # write the classification results and timestamps to a txt file
    with open(os.path.join(results_path,filename[:-4])+".txt", 'w') as fp:
        fp.write(f"speech ratio: {round(speech_ratio, 2)}, disruption ratio: {round(disruption_ratio, 2)}\n")
        for i in range(len(y_labels)):
            fp.write(f"{round(plotter.timestamps[i],2)}, {y_labels[i]}, {round(y_probs[i],2)}\n")

