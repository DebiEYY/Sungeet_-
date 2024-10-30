import numpy as np
import tensorflow as tf
import pandas as pd
import pretty_midi
from prepare_data import midi_to_notes
from keras.models import load_model

# key order for data set
key_order = ['pitch', 'step', 'duration']


# פונקציה שממירה דאטה פריים של תווים לMIDI
def notes_to_midi(
  notes: pd.DataFrame,
  out_file: str,
  instrument_name: str,
  velocity: int = 100,  # note loudness
) -> pretty_midi.PrettyMIDI:

  pm = pretty_midi.PrettyMIDI()
  instrument = pretty_midi.Instrument(
      program=pretty_midi.instrument_name_to_program(
          instrument_name))

  prev_start = 0
  for i, note in notes.iterrows():
    start = float(prev_start + note['step'])
    end = float(start + note['duration'])
    note = pretty_midi.Note(
        velocity=velocity,
        pitch=int(note['pitch']),
        start=start,
        end=end,
    )
    instrument.notes.append(note)
    prev_start = start

  pm.instruments.append(instrument)
  pm.write(out_file)
  return pm


# חיזוי התו הבא בסדרה
def predict_next_note(
        notes: np.ndarray,
        model: tf.keras.Model,
        temperature: float = 1.0) -> tuple[int, float, float]:
    """Generates a note as a tuple of (pitch, step, duration), using a trained sequence model."""

    # זה קנה מידה מסוים שאם משנים אותו התוצאה גם משתנה
    # נראלי פשוט לנסות לשנות ולראות מה יצא
    assert temperature > 0

    # הוספת מימד אצווה
    inputs = tf.expand_dims(notes, 0)

    # החיזוי בעצמו
    predictions = model.predict(inputs)
    pitch_logits = predictions['pitch']
    step = predictions['step']
    duration = predictions['duration']

    # הסרה של מימד האצווה שהוספנו
    # אז למה הוספנו אותו?
    pitch_logits /= temperature
    pitch = tf.random.categorical(pitch_logits, num_samples=1)
    pitch = tf.squeeze(pitch, axis=-1)
    duration = tf.squeeze(duration, axis=-1)
    step = tf.squeeze(step, axis=-1)

    # בדיקה שסטפ ווליו לא שליליים
    # ואם כן אז מחליפה אותם באפס
    step = tf.maximum(0, step)
    duration = tf.maximum(0, duration)

    # ו.... התוצאות
    return int(pitch), float(step), float(duration)


# זאת הפונקציה repeat() !!!
def repeat(midi_path, seq_length, vocab_size):
    temperature = 2.0
    # הגדרת מספר התווים שתחזה
    num_predictions = 120

    df = midi_to_notes(midi_path)

    # המרה של המידי לרשימת תווים
    sample_notes = np.stack([df[key] for key in key_order], axis=1)

    # הרצף הראשוני של הערות; הפיץ' מנורמל בדומה לאימון
    # sequences
    input_notes = (
            sample_notes[:seq_length] / np.array([vocab_size, 1, 1]))

    model_path = r"C:\Users\debie\Desktop\Studies\project\Sungeet_pychm\pythonProject1\models\lstm_model.h5"
    model = load_model(model_path)

    # שומרים את החיזוי ברשימה
    generated_notes = []
    prev_start = 0
    for _ in range(num_predictions):
        # במשתנה מודל אני שולחת לו את הקובץ שנשמר
        pitch, step, duration = predict_next_note(input_notes, model, temperature)
        start = prev_start + step
        end = start + duration
        input_note = (pitch, step, duration)
        generated_notes.append((*input_note, start, end))
        input_notes = np.delete(input_notes, 0, axis=0)
        input_notes = np.append(input_notes, np.expand_dims(input_note, 0), axis=0)
        prev_start = start

    # המרה של רשימת התווים לדאטה פריים
    generated_notes = pd.DataFrame(
        generated_notes, columns=(*key_order, 'start', 'end'))

    out_file = r"C:\Users\debie\Downloads\out_file.midi"

    out_pm = notes_to_midi(
        generated_notes, out_file, instrument_name=instrument_name)

    # לשלוח ללקוח
    return out_pm

    # אולי לשלוח גם ניתוב לא יודעתתתתת
