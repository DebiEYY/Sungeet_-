from midi2audio import FluidSynth
import os

def midi_to_wav(midi_file_path, output_wav_path, soundfont_path = r"C:\soundfonts\FluidR3_GM.sf2"):

    # הכנת נתיב לשמירת קובץ WAV
    # מחלקים את המחרוזת לנקודות
    parts = midi_file_path.split('.')

    # בודקים אם יש יותר מחלק אחד (כדי לוודא שיש נקודה במחרוזת)
    if len(parts) > 1:
        # מאחדים את כל החלקים מלבד האחרון
        output_wav_path = '.'.join(parts[:-1])
    else:
        # אם אין נקודה במחרוזת, משאירים אותה כמו שהיא
        output_wav_path = midi_file_path

    output_wav_path = f'{output_wav_path}.wav'

    # בדיקה אם קובץ soundfont קיים
    if not os.path.exists(soundfont_path):
        raise FileNotFoundError(f"Soundfont file not found: {soundfont_path}")

    # יצירת אובייקט FluidSynth עם soundfont
    fs = FluidSynth(soundfont_path)

    # המרת קובץ MIDI ל-WAV
    fs.midi_to_audio(midi_file_path, output_wav_path)

    # חזרה של נתיב קובץ ה-WAV שנוצר
    return output_wav_path


