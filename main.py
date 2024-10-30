import pretty_midi
from mido import MidiFile
from models.knn import get_genre_by_knn
import json
from backend.server import upload_midi, send_to_app, send_chords_midi_and_txt_links
from models.predict_lstm import repeat
from chords.chords_algo import add_chords

def send_params_in_json(midi, genre):
    pm = pretty_midi.PrettyMIDI(midi)
    instrument = pm.instruments[0]
    instrument_name = pretty_midi.program_to_instrument_name(instrument.program)
    mid = MidiFile(midi, clip=True)
    msg = mid.tracks[0][2]  # מציאת הסולם
    if msg.type == 'key_signature':
        key = msg.key

    bpm = 120
    with open('params.json', 'w', encoding='utf-8') as f:
        json.dump(f"{instrument_name}/n{key}/n{genre}/n{bpm}", f, ensure_ascii=False, indent=4)

    return json


def main():
    # get http from React
    midi = upload_midi()
    # get genre
    genre = get_genre_by_knn(midi)
    # send params
    json_file = send_params_in_json(genre)
    # compose the song
    pm = repeat(midi, 50, 128)
    # add chords
    chords_list = add_chords(midi)
    # send all to client
    send_to_app(pm)
    # send chords if ask
    send_chords_midi_and_txt_links(chords_list)




