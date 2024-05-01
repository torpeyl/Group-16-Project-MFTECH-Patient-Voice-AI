import sounddevice as sd
import re

filename = 'data/voice001.txt'

with open(filename) as f:
    li = f.readlines()
    li = [re.sub(r'[\n]', '', s) for s in li]
    li = [float(i) for i in li]

sd.play(li, 8000, blocking=True, loop=False)
sd.stop()