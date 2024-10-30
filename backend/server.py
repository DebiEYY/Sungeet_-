from flask import Flask, request, jsonify
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)


@app.route('/upload_midi', methods=['POST'])
def upload_midi():
    data = request.get_json()
    midi_path = data.get('midi_path')
    if midi_path:
        # Process the MIDI file path as needed
        print(f"Received MIDI path: {midi_path}")
        return jsonify({"message": "MIDI path received successfully"}), 200
    else:
        return jsonify({"error": "No MIDI path provided"}), 400


@app.route('/adding', methods=['GET'])
def get_string():
    global chords

    try:
        # קריאה מתוך קובץ ה- JSON הראשי
        with open('chords.json', 'r', encoding='utf-8') as f:
            chords_links = json.load(f)

        # יצירת רשימה עבור התוצאות של ה־diagnosis בקובץ
        diagnosis_messages = [result['diagnosis'] for result in chords_links]

        # יצירת מחרוזת מהתוצאות
        formatted_diagnoses = "".join(diagnosis_messages)

        # הדפסת ה־diagnosis מכל תוצאה ב־JSON
        print("Diagnoses:")
        for diagnosis in diagnosis_messages:
            print(diagnosis)

        # חזרה עם תשובה ל־JSON עם ירידות שורה
        return jsonify({'message': formatted_diagnoses}), 200

    except Exception as e:
        print('Error:', str(e))
        return jsonify({'error': 'An error occurred while retrieving diagnoses'}), 500


def send_to_app():
    pass


def send_chords_midi_and_txt_links():
    pass


if __name__ == '__main__':
    app.run(port=5000, debug=True)


