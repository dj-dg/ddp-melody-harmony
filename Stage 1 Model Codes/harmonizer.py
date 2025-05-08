import os
import numpy as np
from music21 import *
from loader import get_filenames, convert_files
from model import build_model
from config import *
from tqdm import trange
from copy import deepcopy

# use cpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

chord_dictionary = ['R',
    'Cm', 'C',
    'C#m', 'C#',
    'Dm', 'D',
    'D#m', 'D#',
    'Em', 'E',
    'Fm', 'F',
    'F#m', 'F#',
    'Gm', 'G',
    'G#m', 'G#',
    'Am', 'A',
    'A#m', 'A#',
    'Bm', 'B'
]

def predict(song, model):
    chord_list = []
    # Process in 8 half-measure chunks
    for idx in range(int(len(song)/8)):
        melody = [song[idx*8 + i] for i in range(8)]
        melody = np.array([np.array(seg, dtype=np.float32) for seg in melody])[np.newaxis, ...]
        net_output = model.predict(melody)[0]
        for chord_idx in net_output.argmax(axis=1):
            chord_list.append(chord_dictionary[chord_idx])
    # Handle remaining half-measures
    remaining = len(song) % 8
    if remaining > 0:
        melody = song[-remaining:] + [np.zeros(12, dtype=np.float32)]*(8-remaining)
        melody = np.array([np.array(seg, dtype=np.float32) for seg in melody])[np.newaxis, ...]
        net_output = model.predict(melody)[0]
        for idx in range(remaining):
            chord_list.append(chord_dictionary[net_output[idx].argmax()])
    return chord_list

# def export_music(score, chord_list, gap_list, filename):
#     from music21 import harmony, note, stream
#     import os

#     new_score = deepcopy(score)
    
#     # Remove existing chord symbols
#     for cs in new_score.recurse().getElementsByClass(harmony.ChordSymbol):
#         cs.activeSite.remove(cs)

#     m_idx = 0
#     for m in new_score.recurse().getElementsByClass('Measure'):
#         if m_idx+1 >= len(chord_list):
#             break
            
#         measure_duration = m.duration.quarterLength
#         current_chord1 = chord_list[m_idx]
#         current_chord2 = chord_list[m_idx+1]

#         # Insert first chord
#         if current_chord1 != 'R':
#             cs1 = harmony.ChordSymbol(current_chord1)
#             cs1.offset = 0.0
#             cs1.writeAsChord = False
            
#             # Check if both chords are same
#             if current_chord1 == current_chord2:
#                 cs1.duration.quarterLength = measure_duration  # Full measure
#                 m.insert(0.0, cs1)
#                 m_idx += 1  # Skip next chord
#                 continue
#             else:
#                 cs1.duration.quarterLength = measure_duration / 2.0
#                 m.insert(0.0, cs1)

#         # Insert second chord if different
#         if current_chord2 != 'R' and current_chord1 != current_chord2:
#             cs2 = harmony.ChordSymbol(current_chord2)
#             cs2.offset = measure_duration / 2.0
#             cs2.duration.quarterLength = measure_duration / 2.0
#             cs2.writeAsChord = False
#             m.insert(measure_duration / 2.0, cs2)

#         m_idx += 2

#     # Export
#     os.makedirs(OUTPUTS_PATH, exist_ok=True)
#     output_path = os.path.join(OUTPUTS_PATH, f"{os.path.basename(filename)}.mxl")
#     new_score.write('musicxml', fp=output_path)

def export_music(score, chord_list, gap_list, filename):
    from music21 import harmony, note, stream
    new_score = deepcopy(score)
    
    # Remove existing chords
    for cs in new_score.recurse().getElementsByClass(harmony.ChordSymbol):
        cs.activeSite.remove(cs)

    m_idx = 0
    for m in new_score.recurse().getElementsByClass('Measure'):
        if m_idx+1 >= len(chord_list):
            break
            
        # Get transposition intervals for this measure
        gap1 = gap_list[m_idx]
        gap2 = gap_list[m_idx+1]

        # First half chord
        if chord_list[m_idx] != 'R':
            cs1 = harmony.ChordSymbol(chord_list[m_idx])
            cs1 = cs1.transpose(-gap1.semitones)
            cs1.quarterLength = m.duration.quarterLength / 2
            cs1.writeAsChord = False
            m.insert(0.0, cs1)

        # Second half chord (if different)
        if (m_idx+1 < len(chord_list) and 
            chord_list[m_idx+1] != 'R' and 
            chord_list[m_idx+1] != chord_list[m_idx]):
            cs2 = harmony.ChordSymbol(chord_list[m_idx+1])
            cs2 = cs2.transpose(-gap2.semitones)
            cs2.quarterLength = m.duration.quarterLength / 2
            cs2.writeAsChord = False
            m.insert(m.duration.quarterLength/2, cs2)

        m_idx += 2

    # Export
    os.makedirs(OUTPUTS_PATH, exist_ok=True)
    output_path = os.path.join(OUTPUTS_PATH, f"{os.path.basename(filename)}.mxl")
    new_score.write('musicxml', fp=output_path)




if __name__ == '__main__':
    model = build_model(weights_path=WEIGHTS_PATH)
    filenames = get_filenames(input_dir=INPUTS_PATH)
    data_corpus = convert_files(filenames, fromDataset=False)
    for idx in trange(len(data_corpus)):
        melody_vecs = data_corpus[idx][0]
        gap_list = data_corpus[idx][1]
        score = data_corpus[idx][2]
        filename = data_corpus[idx][3]
        chord_list = predict(melody_vecs, model)
        export_music(score, chord_list, gap_list, filename)
