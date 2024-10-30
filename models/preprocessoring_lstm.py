# lstm
from pretty_midi import PrettyMIDI
import tensorflow as tf
import pathlib
import prepare_data


def read_midi(path):
    pm = PrettyMIDI(path)
    return pm


def mse_with_positive_pressure(y_true: tf.Tensor, y_pred: tf.Tensor):
    mse = (y_true - y_pred) ** 2
    positive_pressure = 10 * tf.maximum(-y_pred, 0.0)
    return tf.reduce_mean(mse + positive_pressure)


def build_model(seq_length):
  # באיזה צורה הנתונים נכנסים למודל
  input_shape = (seq_length, 3)
  # הגדרת קצב הלמידה
  learning_rate = 0.005

  # מודל LSTM!!!!!!!!!!!!!!!!!!!
  # יצירת שכבת קלט- שכבת ה-input
  inputs = tf.keras.Input(input_shape)
  # שכבת LSTM עם 128 יחידות חישוב- נוירונים (כי יש 128 תווים)
  x = tf.keras.layers.LSTM(128)(inputs)

  # שלוש שכבות פלט עבור המודל: פיץ', סטפ, משך
  outputs = {
    'pitch': tf.keras.layers.Dense(128, name='pitch')(x),
    'step': tf.keras.layers.Dense(1, name='step')(x),
    'duration': tf.keras.layers.Dense(1, name='duration')(x),
  }

  # יצירת המודל לפי ההגדרות
  model = tf.keras.Model(inputs, outputs)

  # הגדרת פונקציית LOSS
  # לכל אחד מהפלטים של המודל:
  # עבור התו עצמו- לפי המודל
  # עבור משך וחוזק בשימוש בפונקציה המותאמת
  loss = {
        'pitch': tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True),
        'step': mse_with_positive_pressure,
        'duration': mse_with_positive_pressure,
  }

  # בחירת אופטימציה למודל
  # אופטימציה אל אדם משמשת למזער את פונקציית השגיאה במהלך האימון
  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

  # קומפילציה של המודל עם הפונקציות איבוד ואופטימיזר שהוגדרו
  model.compile(loss=loss,
                loss_weights={
                'pitch': 0.05,
                'step': 1.0,
                'duration': 1.0, },
                 optimizer=optimizer)

  # הדפסת כל השכבות במודל כולל שם וסוג וכן צורת פלט עבור כל שכבה
  model.summary()

  return model


def preprocessing():
    model = build_model(100)
    epochs = 30
    callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath='./training_checkpoints/ckpt_{epoch}.weights.h5',
        save_weights_only=True),
    # קביעת אחוזי עצירה מוקדמת עקב אי התקדמות או אובר פיטינג
    tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=5,
        verbose=1,
        restore_best_weights=True),
    ]
    return model, epochs, callbacks


def train_model():
    # ניתוב לתיקית הנתונים
    data_dir = pathlib.Path('C://Users//debie//Desktop//Studies//project//maestro-v3.0.0-midi//maestro-v3.0.0')
    # קריאת כל הקבצים שיש בדטה הזה
    filenames = prepare_data.read_csv(data_dir)
    # הכנה של הדטה להכנסה למודל
    train_ds = prepare_data.prepare_dataset(filenames, 75, 128, 64)
    # שמירת נתונים של המודל
    model, epochs, callbacks = preprocessing()
    # אימון המודל
    model.fit(
        train_ds,
        epochs=epochs,
        callbacks=callbacks,
        )
    # שמירת המודל
    model.save(f"lstm_model.h5")
    # זהו

