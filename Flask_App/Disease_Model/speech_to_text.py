import speech_recognition as sr


def speech2text(file_path):
    r = sr.Recognizer()

    myaudio = sr.AudioFile(file_path)

    with myaudio as source:
        audio = r.record(source)

    output_string = r.recognize_google(audio)
    print(output_string)
    return output_string

file_path = "./downloads/recording.wav"

test = speech2text(file_path)