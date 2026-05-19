import os
import subprocess
import numpy as np
import pandas as pd
import librosa
import tensorflow_hub as hub
from static_ffmpeg import add_paths

add_paths()

model = hub.load('https://tfhub.dev/google/yamnet/1')
class_map_path = model.class_map_path().numpy()
class_names = [line.split(',')[2].strip('"') for line in open(class_map_path).readlines()[1:]]

target_sounds = ['Music', 'Singing', 'Musical instrument', 'Child singing', 'Humming', 'Lullaby']
base_dir = './corpus'
output_dir = './clips'
if not os.path.exists(output_dir): os.makedirs(output_dir)

results = []

for child in ['Tosya', 'Yasha']:
    path = os.path.join(base_dir, child)
    if not os.path.exists(path): continue
    files = [f for f in os.listdir(path) if f.lower().endswith(('.mp4', '.avi', '.mts'))]

    for f in files:
        try:
            y, sr = librosa.load(os.path.join(path, f), sr=16000)
            scores, _, _ = model(y)
            scores_np = scores.numpy()

            for i, frame in enumerate(scores_np):
                label = class_names[np.argmax(frame)]
                conf = np.max(frame)
                time = i * 0.48

                s_s = int(time * sr)
                e_s = int((time + 0.48) * sr)
                y_h, y_p = librosa.effects.hpss(y[s_s:e_s])
                m_score = np.sum(y_h**2) / (np.sum(y_p**2) + 1e-6)

                if (label in target_sounds and conf > 0.15) or (label == 'Child speech' and m_score > 2.5):
                    results.append([child, f, time, label, m_score])
        except:
            continue

df = pd.DataFrame(results, columns=['Child', 'File', 'Start', 'Label', 'Score'])

final_list = []
for (child, file), group in df.groupby(['Child', 'File']):
    group = group.sort_values('Start')
    if group.empty: continue
    s_start = group.iloc[0]['Start']
    s_last = s_start
    for idx in range(1, len(group)):
        curr = group.iloc[idx]['Start']
        if curr - s_last <= 15:
            s_last = curr
        else:
            final_list.append([child, file, s_start, s_last + 2])
            s_start = curr
            s_last = curr
    final_list.append([child, file, s_start, s_last + 2])

for res in final_list:
    child, file, start, end = res
    dur = end - start
    if dur < 2.5: continue
    out = os.path.join(output_dir, f"{child}_{int(start)}_{file}")
    inp = os.path.join(base_dir, child, file)
    cmd = ['ffmpeg', '-y', '-ss', str(start), '-t', str(dur), '-i', inp, 
           '-c:v', 'libx264', '-preset', 'ultrafast', '-c:a', 'aac', out]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
