import mido
import data_structures


def read_midi(midi_file):
    return mido.MidiFile(midi_file, clip=True)


all_notes = data_structures.CircularList()
all_notes.append('C',  60)
all_notes.append('C#', 61)
all_notes.append('D',  62)
all_notes.append('D#', 63)
all_notes.append('E',  64)
all_notes.append('F',  65)
all_notes.append('F#', 66)
all_notes.append('G',  67)
all_notes.append('G#', 68)
all_notes.append('A',  69)
all_notes.append('A#', 70)
all_notes.append('B',  71)

# global variable no chord

no_chord = data_structures.Chord("no_chord", 0)
for _ in range(3):
    no_chord.add_note('z', 0)


###################
# build chords#
###################

def find_scale(mid):
   msg = mid.tracks[0][2]  # מציאת הסולם של המנגינה
   if msg.type == 'key_signature':
            return msg.key


def get_degree_note(all_notes, scale, degree):
    current = all_notes.head  # מתחילים מהראש של הרשימה המעגלית
    while current.data1 != scale:
        current = current.next  # מתקדמים לצומת הבאה ברשימה עד שמוצאים את התו הראשון של הסולם

    for _ in range(degree - 1):  # מעבירים את current להיות הצומת המתאימה לדרגה
        current = current.next

    return current.data1  # מחזירים את הערך של התו במקום המבוקש


def build_chord(all_notes, scale, degree):
    chord = data_structures.Chord(scale, degree)

    # מציאת התו ההתחלתי המתאים לסולם
    start = all_notes.head
    while start.data1 != scale:
        start = start.next
        if start == all_notes.head:
            return chord  # return an empty chord if no matching note is found

    # הוספת התו ההתחלתי לאקורד
    chord.add_note(start.data1, start.data2)

    # הוספת התווים במיקומים 4 ו-7
    for i in [4, 7]:
        current = start
        for _ in range(i):
            current = current.next
        chord.add_note(current.data1, current.data2)

    return chord


def primary_chords_of_scale(all_notes, mid):
    scale = find_scale(mid)
    primary_chords = []
    degrees = [1, 6, 8]

    for degree in degrees:
        start_note = get_degree_note(all_notes, scale, degree)
        chord = build_chord(all_notes,start_note, degree)
        primary_chords.append(chord)
    return primary_chords


# end of build chords


###################
# split to bars #
###################


# מציאת הטמפו של השיר
def get_tempo(mid):
    for msg in mid.tracks[0]:
        if msg.type == 'set_tempo':
            return msg.tempo
    return 500000  # ברירת מחדל: 120 BPM


# פונקצית עזר להמרה של טיק לשניה לפי הביט וה-ticks_per_beat
def ticks_to_seconds(ticks, ticks_per_beat, tempo):
    # טמפו במיקרו שניות לפעימה אחת
    microseconds_per_beat = tempo # כמה ביטים יש במיקרו שניה
    seconds_per_beat = microseconds_per_beat / 1_000_000 # המרה ממיקרו שניה לשניה
    return (ticks / ticks_per_beat) * seconds_per_beat #  החזרה של המרה מטיקים למילישניות
# חלוקה של מספר הטיקים הנוכחי במספר הטיקים לביט ואז הכפלה במספר המילישניות לביט


# חישוב המשך של כל תו בשניות כולל חישוב דמימות
def note_durations_in_seconds(mid):
    ticks_per_beat = mid.ticks_per_beat
    tempo = get_tempo(mid)

    note_durations = []

    for track in mid.tracks:

        # תעבור על כל התווים ברצועה
        for msg in track:

            # אם זה הודעה של נגן תו וזה דמימה
            if msg.type == 'note_on' and msg.velocity > 0 and msg.time > 1:
                # שמור במשתנה דמימה
                rest = msg.time
                # המרה לשניות
                rest = ticks_to_seconds(rest, ticks_per_beat, tempo)
                # שמירת הדמימה עם תו 0
                note_durations.append((0, rest))

            # אם זה הודעה של תפסיק לנגן תו
            elif msg.type == 'note_on' and msg.velocity == 0 or msg.type == 'note_off':
                duration = msg.time
                # המרה לשניות של משך התו
                duration = ticks_to_seconds(duration, ticks_per_beat, tempo)
                # שמירת התו והמשך שלו בשניות
                note_durations.append((msg.note, duration))

    return note_durations


# המרה של המשכים לערך הקרוב ביותר במונחים מוזיקליים
def get_note_length_in_musical_terms(duration_in_seconds, tempo):
    if tempo is None:
        tempo = 500000  # ברירת מחדל: 120 BPM
    seconds_per_beat = tempo / 1_000_000  # המרה למילישניות

    # משכים מוזיקליים (בפעימות)
    lengths_in_beats = [4, 2, 1, 0.5, 0.25, 0.125, 0.0625, 0.03125]

    # מציאת המשך הקרוב ביותר
    closest_length = None
    min_difference = float('inf')  # ערך התחלתי גבוה מאוד כדי להבטיח שכל הבדל יהיה קטן ממנו

    # מחשבים את המשך בביטים
    duration_in_beats = duration_in_seconds / seconds_per_beat

    # לולאה על כל ערך ב-lengths_in_beats
    for length in lengths_in_beats:
        difference = abs(duration_in_beats - length)
        if difference < min_difference:
            min_difference = difference
            closest_length = length

    return closest_length


# חישוב המשכים מוזיקליים
def duration_in_musical_terms(mid):
    musical_list = []
    tempo = get_tempo(mid)
    note_durations_in_seconds_list = note_durations_in_seconds(mid)
    for note, duration_in_seconds in note_durations_in_seconds_list:
        length_in_musical_terms = get_note_length_in_musical_terms(duration_in_seconds, tempo)
        musical_list.append([note, length_in_musical_terms])
        # print(f"Note {note} duration in musical terms: {length_in_musical_terms}")
    return musical_list


def get_time_signature(mid):
    for msg in mid.tracks[0]:
        if msg.type == 'time_signature':
            return msg.numerator, msg.denominator


def split_to_bars(mid):
    list_of_bars = []
    bar = []
    current_sum = 0

    musical_list = duration_in_musical_terms(mid)
    numerator, denominator = get_time_signature(mid)

    for note, duration in musical_list:
        if current_sum + duration <= numerator:
            bar.append((note, duration))
            current_sum += duration
        else:
            list_of_bars.append(bar)
            bar = [(note, duration)]
            current_sum = duration

        if current_sum == numerator:
            list_of_bars.append(bar)
            bar = []
            current_sum = 0

    if bar:
        list_of_bars.append(bar)

    return list_of_bars

# end of split to bars

