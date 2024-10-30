from collections import Counter
from chords.preprocessing import no_chord, split_to_bars, primary_chords_of_scale, all_notes
from mido import MidiFile, MidiTrack, Message
import json


def adjust_note_to_range(note, min_range=60, max_range=71):
    if note == 0:
        return note
    while note < min_range:
        note += 12
    while note > max_range:
        note -= 12
    return note


def compare_chord_to_bar(chord, bar):
    # חילוץ הפיץ של האקורדים
    chord_pitch = []
    for pitch in chord.notes:
        chord_pitch.append(pitch[1])  # o(3)

    # התאמת התווים בתיבה לטווח הנדרש
    adjusted_bar = [(pitch if pitch == 0 else adjust_note_to_range(pitch), duration) for pitch, duration in bar]

    # ספירה כמה פעמים מופיע כל תו בתיבה
    note_count = Counter(pitch for pitch, _ in adjusted_bar)  # o(n)

    # בדיקה כמה תווים מתאימים בין התיבה לאקורד
    sum = 0
    for note in chord_pitch:
        if note_count[note]:
            sum += 1
    return sum


def loop_note(chord, bar):
    chord_pitches = [note[1] for note in chord.notes]  # שליפת הפיצ'ים של האקורד

    # התאמת התווים בתיבה לטווח הנדרש
    adjusted_bar = [(adjust_note_to_range(pitch), duration) for pitch, duration in bar]

    print("bar before: ", bar)
    print("bar after: ", adjusted_bar)

    # ספירה כמה פעמים מופיע כל תו בתיבה
    note_count = Counter(pitch for pitch, _ in adjusted_bar)  # o(n)

    if not note_count:
        return 0

    # מצא את הערך המקסימלי
    max_count = max(note_count.values())

    # מציאת כל התווים שהערך שלהם שווה לערך המקסימלי
    max_notes = [note for note, count in note_count.items() if count == max_count]
    max_pitch = max_notes[0]

    #  # או שיש רק תו אחד בתיבה          אם יש אקורד שזה השם שלו- תשים אותו
    if chord_pitches[0] == max_pitch and max_count > 2 or chord_pitches[0] == max_pitch and len(note_count) == 1:
        return 3

    # אם לא - תבדוק באיזה אקורד יש הכי הרבה התאמה
    elif max_count >= 2 and max_pitch in chord_pitches:
        return 3

    # אם אין התאמה עדיין
    return 0


def add_chords(mid):
    # הכנה של השיר:
    list_of_bars = split_to_bars(mid)
    chords = primary_chords_of_scale(all_notes, mid)

    # יוצרים מילון שישמור את רמת ההתאמה לכל אקורד
    chords_cnt = {chord: 0 for chord in chords}
    # רשימת האקורדים של השיר
    chords_for_song = []

    # בדיקה על כל התיבות
    for i in range(len(list_of_bars)):
        # דגל שבודק אם התאמנו כבר אקורד
        found_chord = False
        print("----new itration----")
        print("bar number: ", i)
        # ניסיון להתאים עם תיבה אחת
        for chord in chords:
            # השוואה בין האקורד לתיבה
            sum_result = compare_chord_to_bar(chord, list_of_bars[i])
            # שמירה כמה תווים התאימו בין האקורד לתיבה
            chords_cnt[chord] = sum_result
            print("sum_result: ", sum_result)
        # צא מלולאה
        # בדוק למי התאמה הכי גבוהה
        match = max(chords_cnt, key=chords_cnt.get)
        max_match = chords_cnt[match]
        print("max_match: ", max_match)
        # בדוק כמה התאמות יש
        max_chords = [chord for chord, count in chords_cnt.items() if count == max_match]

        # אם יש יותר מ2 התאמות
        if len(max_chords) >= 2:
            #
            # מילון נוסף
            best = {c1: 0 for c1 in max_chords}
            # תבדוק כפילויות של תווים בתיבה
            for c in max_chords:
                print("#####################")
                print("the chord: ", c.name)
                print("the bar: ", list_of_bars[i])
                best[c] = (loop_note(c, list_of_bars[i]))
            # תשמור את הטוב ביותר
            match = max(best, key=best.get)
            # ועד כמה הוא היה טוב
            loop_max = best[match]
            print("loop_max: ", loop_max)
            if loop_max != 0:
                # הוסף אקורד
                chords_for_song.append(match)
                # סמן שהוספת
                found_chord = True
                print(f"Added {match.name} from single bar at bar {i} by loop_note")

        if max_match >= 2 and len(max_chords) <= 1:
            # הוסף אקורד
            chords_for_song.append(match)
            # סמן שהוספת
            found_chord = True
            # debug
            print(f"Added {match.name} from single bar at bar {i}")

        # אם לא מצאת שום אקורד מתאים לתיבה הזאת
        elif not found_chord:
            # שים Placeholder
            chords_for_song.append(no_chord)
            # debug
            print(f"No Match found at bar {i}")

    # תחזיר את רשימת האקורדים שמצאת
    return chords_for_song


###################
# save the results #
###################


def create_txt_file(chords, filename):
    filename = f"{filename}\your_chords.txt"
    with open(filename, 'w') as file:
        for i, chord in enumerate(chords):
            file.write(f"bar num {i},  the chord: {chord.name}\n")
    return filename



def create_chords_midi(filename, chords):
    filename = f"{filename}\your_chords.midi"
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    chord_pitches = []

    # ספירת טיקים לכל פעימה
    ticks_per_beat = mid.ticks_per_beat

    # יצירת רשימת תווים של האקורדים
    for chord in chords:
        chord_pitches.append([note[1] for note in chord.notes])  # שליפת הפיצ'ים של האקורד

    # יצירת תווים בקובץ MIDI עבור כל אקורד
    for chord_notes in chord_pitches:
        for note in chord_notes:
            track.append(Message('note_on', note=note, velocity=64, time=0))
        track.append(Message('note_off', note=chord_notes[0], velocity=64, time=ticks_per_beat))  # סגירת אקורד

    mid.save(filename)
    return filename


def export_chords(filename=r"C:\Users\debie\Downloads", chords_for_song=[]):
    txt = create_txt_file(filename=filename, chords=chords_for_song)
    midi = create_chords_midi(filename=filename, chords=chords_for_song)
    with open('chords.json', 'w', encoding='utf-8') as f:
        json.dump(f"{txt}/n{midi}", f, ensure_ascii=False, indent=4)
