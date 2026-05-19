import os
import csv
import numpy as np
import librosa
from static_ffmpeg import add_paths

add_paths()

print("--- ЗАПУСК ГАРМОНИЧЕСКОГО ДЕТЕКТОРА (ПОИСК МЕЛОДИЙ) ---")
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(BASE_PATH, 'MUSICAL_MOMENTS_FOUND.csv')

def format_time(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"

with open(OUTPUT_FILE, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Child', 'File', 'Start', 'End', 'Duration', 'Melodic_Score'])

    for child in ['Yasha', 'Tosya']:
        folder_path = os.path.join(BASE_PATH, child)
        if not os.path.exists(folder_path): continue
        
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.mts'))]
        
        for file_name in files:
            print(f"Анализ: {child} -> {file_name}")
            try:
                y, sr = librosa.load(os.path.join(folder_path, file_name), sr=16000)
                if len(y) == 0: continue
                
                # Отделяем чистые тона (гармоники) от шума
                y_harm, y_perc = librosa.effects.hpss(y)
                
                # Считаем энергию по секундам
                step = 16000
                h_energy = np.array([np.sum(y_harm[i:i+step]**2) for i in range(0, len(y_harm), step)])
                p_energy = np.array([np.sum(y_perc[i:i+step]**2) for i in range(0, len(y_perc), step)])
                
                # Коэффициент певучести
                ratios = h_energy / (p_energy + 0.0001)
                
                cur = None
                found = 0
                
                for i, ratio in enumerate(ratios):
                    if ratio > 2.5: # Если мелодия в 2.5 раза чище шума
                        if cur is None:
                            cur = {'start': i, 'max': ratio}
                        cur['end'] = i + 1
                        cur['max'] = max(cur['max'], ratio)
                    else:
                        if cur:
                            dur = cur['end'] - cur['start']
                            if dur >= 2:
                                writer.writerow([child, file_name, format_time(cur['start']), format_time(cur['end']), dur, round(float(cur['max']), 1)])
                                found += 1
                            cur = None
                f.flush()
                print(f"   Найдено фрагментов: {found}")
            except Exception as e:
                print(f"   Ошибка в {file_name}: {e}")

print(f"\nАНАЛИЗ ЗАВЕРШЕН! Файл: {OUTPUT_FILE}")
input("Нажмите Enter...")