# lstm
import collections
import fluidsynth
import glob
import numpy as np
import pandas as pd
import pretty_midi
from pretty_midi import PrettyMIDI
import tensorflow as tf
import pathlib


###################
# global variables#
###################

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# Sampling rate for audio playback
_SAMPLING_RATE = 16000

# key order for data set
key_order = ['pitch', 'step', 'duration']


# נתיב לתיקיית הנתונים
data_dir = pathlib.Path('C://Users//debie//Desktop//Studies//project//maestro-v3.0.0-midi//maestro-v3.0.0')


def read_csv(path):
    # בדיקה אם התיקייה קיימת
    if not data_dir.exists():
        print("תיקיית הנתונים לא קיימת.")
    else:
        # קריאת קבצים מהתיקייה (לדוגמה)
        for file_path in data_dir.iterdir():
            print("קובץ מסוג:", file_path.name)

    # קריאת כל קבצי MIDI בתיקיות של הדאטה סט
    filenames = glob.glob(str(data_dir / '**/*.mid*'))
    print('Number of files:', len(filenames))

    return filenames


#המרה של הקובץ לתווים
def midi_to_notes(midi_file: str) -> pd.DataFrame:
  pm = PrettyMIDI(midi_file)
  instrument = pm.instruments[0]
  notes = collections.defaultdict(list)

  # Sort the notes by start time
  sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
  prev_start = sorted_notes[0].start

  for note in sorted_notes:
    start = note.start
    end = note.end
    notes['pitch'].append(note.pitch)
    notes['start'].append(start)
    notes['end'].append(end)
    notes['step'].append(start - prev_start)
    notes['duration'].append(end - start)
    prev_start = start

  return pd.DataFrame({name: np.array(value) for name, value in notes.items()})


def create_tf_dataset(filenames):
    # משתנה ששומר את התווים
    all_notes = []
    # עוברים על כל הקבצים ומחלצים מכל אחד את התווים שלו
    for f in filenames:
        notes = midi_to_notes(f)
        all_notes.append(notes)

    # יצירת דטה פריים מרשימת התווים
    all_notes = pd.concat(all_notes)

    # המרה של הדטה פריים למחסנית של נם פאי
    train_notes = np.stack([all_notes[key] for key in key_order], axis=1)

    # סופסוף המרה לדטה סט של טנסור פלואו
    notes_ds = tf.data.Dataset.from_tensor_slices(train_notes)

    return notes_ds


# פונקציה שמכינה את הדאטה להכנסה למודל
# בסוגריים יש את הערכים שהפונקציה מקבלת:
# דטה סט של טנסור
# אורך סדרה
# מה הטווח של התוצאות- 0-128
# והיא מחזירה אובייקט מסוג דטה סט של טנסור שמכיל רצפי קלט ותגיות
def create_sequences(
        dataset: tf.data.Dataset,
        seq_length: int,
        vocab_size=128,
) -> tf.data.Dataset:
    """Returns TF Dataset of sequence and label examples."""

    # אורך הסדרה ועוד 1 בשביל התווית
    seq_length = seq_length + 1

    # יוצרים חלונות על הנתונים
    # היא מקבלת את הנתונים האלה:
    # seq_length - גודל החלון, כאן שמים את אורך הסדרה
    # shift - כמות הזזה בין חלונות עוקבים- זה אומר כמה תוים לזוז בכל באץ
    # stride - המרווח בין הנתונים בחלון
    # drop_remainder - אם לשמור רק חלונות מלאים- כן
    windows = dataset.window(seq_length, shift=1, stride=1,
                             drop_remainder=True)

    #
    # `flat_map` flattens the" dataset of datasets" into a dataset of tensors
    flatten = lambda x: x.batch(seq_length, drop_remainder=True)
    sequences = windows.flat_map(flatten)

    # Normalize note pitch
    # מנרמלים את הפיץ' של התווים לערך בין -1 ל-1 כדי שהמודל יוכל להתמודד איתם
    def scale_pitch(x):
        x = x / [vocab_size, 1.0, 1.0]
        return x

    # חלוקה לתוויות
    def split_labels(sequences):
        # כל התווים חוץ מהאחרון הם המידע
        inputs = sequences[:-1]
        # התו האחרון הוא התווית
        labels_dense = sequences[-1]
        labels = {key: labels_dense[i] for i, key in enumerate(key_order)}
        # מחזיר
        return scale_pitch(inputs), labels

    return sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)


# הכנה של הדטה להכנסה למודל
def prepare_dataset(filenames, seq_length=100, vocab_size=128, batch_size=64):

    notes_ds, n_notes = create_tf_dataset(filenames, key_order)

    # יוצרים סדרה לאימון המודל לפי המשתנים שהגדרנו
    seq_ds = create_sequences(notes_ds, seq_length, vocab_size)

    # מגדירים את גודל האצוות- זה אומר כמה תווים נשארו עד לסוף המנגינה
    # וזה אומר כמה דוגמאות נכנסות למודל בכל איטרציה במהלך האימון
    buffer_size = n_notes - seq_length

    # שהכנו קודם seq_ds יצירת דטה סט לאימון מבוסס על
    train_ds = (seq_ds
                # שאפל זה לערבב. מערבבים את כל הדוגמאות בדטה סט
                .shuffle(buffer_size)
                # מחלקים את הדטה לבאצים של 64 אחיד
                .batch(batch_size, drop_remainder=True)
                # שומרים את הדטה בקאש כדי לשפר ביצועים ולהפחית זמן קריאה מהדיסק
                .cache()
                # שיפורי ביצועים: בזמן שמודל מעבד את הסדרה הראשונה, הסדרה שניה נטענת
                .prefetch(tf.data.experimental.AUTOTUNE))

    return train_ds



