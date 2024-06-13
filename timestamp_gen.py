# Import the AudioSegment class for processing audio and the 
# split_on_silence function for separating out silent chunks.
from pydub import AudioSegment
from pydub import silence
from pydub.playback import play
import os

# def split(filepath):
#     sound = AudioSegment.from_wav(filepath)
#     chunks = silence.split_on_silence(
#         sound,
#         # min_silence_len = 500,
#         silence_thresh = sound.dBFS - 16,
#         keep_silence = 250, # optional
#     )

def get_silent_timestamps(filepath):
    sound = AudioSegment.from_wav(filepath)
    # play(sound)

    segments = silence.detect_silence(
        sound,
        min_silence_len=500,
        silence_thresh = sound.dBFS - 16    # dependent on background noise level
        )

    # listen to the segments
    counter = 0
    for seg in segments:
        print("No.", counter)
        counter += 1
        sound_seg = sound[seg[0]:seg[1]]
        play(sound_seg)
    
    return segments

def get_nonsilent_timestamps(filepath):
    sound = AudioSegment.from_wav(filepath)
    # play(sound)
    # print(sound.dBFS)

    segments = silence.detect_nonsilent(
        sound,
        min_silence_len=500,
        silence_thresh = sound.dBFS - 16    # dependent on background noise level
        )
    
    return segments


import whisper_timestamped as whisper
import json

def _json_to_segments(json_data):
    data = json.loads(json_data)
    segments = []
    for d in data["segments"]:
        segments.append([int(d["start"]*1000), int(d["end"]*1000)])       # get timestamps in milliseconds
    return segments

def get_speech_timestamps(filepath):
    audio = whisper.load_audio(filepath)
    model = whisper.load_model("tiny", device="cpu")
    transcription = whisper.transcribe(model, audio, language="en")

    data = json.dumps(transcription, indent = 2, ensure_ascii = False)
    with open('transcription.json', 'w', encoding='utf-8') as f:
        f.write(data)
    
    segments = _json_to_segments(data)
    return segments


if __name__ == '__main__':
    directory = "data\\cough-speech-sneeze"
    label = "mixed"
    filename = "qPo0bymnsh0.wav"
    filepath = os.path.join(directory, label, filename)

    # print(get_nonsilent_timestamps(filepath))
    print(get_speech_timestamps(filepath))

