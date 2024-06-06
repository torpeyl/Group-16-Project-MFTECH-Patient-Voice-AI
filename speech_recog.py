import speech_recognition as sr

filepath = 'recordings/harvard.wav'

r = sr.Recognizer()

myaudio = sr.AudioFile(filepath)

with myaudio as source:
    audio = r.record(source)

print(r.recognize_google(audio))