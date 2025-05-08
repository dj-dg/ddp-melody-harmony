import os
import numpy as np
from music21 import *
from loader import get_filenames, convert_files
from model import build_model
from config import *
from tqdm import trange
from copy import deepcopy
import random

SEVENTH_PROBABILITY = 0.8  # 80% chance for 7th conversion
JAZZ_SUB_PROB = 0.7  # 30% chance per eligible chord



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

def convert_to_seventh_chords(chord_list, prob=0.5):
    """
    Randomly convert chords to 7ths based on probability
    - prob: 0.0 (never) to 1.0 (always)
    - 'R' remains unchanged regardless of probability
    """
    seventh_chord_list = []
    for chord in chord_list:
        if chord == 'R' or random.random() > prob:
            seventh_chord_list.append(chord)
        else:
            seventh_chord_list.append(chord + ('7' if not chord.endswith('m') else '7'))
    return seventh_chord_list


def jazzify_chords(chord_list, sub_prob=0.3):
    """
    Applies jazz transformations:
    - 30% chance for tritone substitution on dominant 7ths
    - Adds 9th extensions to 20% of 7th chords
    """
    jazz_chords = []
    tritone_map = {
        'C': 'F#', 'C#': 'G', 'D': 'G#', 'D#': 'A',
        'E': 'A#', 'F': 'B', 'F#': 'C', 'G': 'C#',
        'G#': 'D', 'A': 'D#', 'A#': 'E', 'B': 'F'
    }
    
    for chord in chord_list:
        if chord == 'R':
            jazz_chords.append(chord)
            continue
            
        # Tritone substitution for dominant 7ths
        if chord.endswith('7') and not chord.endswith('m7'):
            root = chord[:-1]
            if random.random() < sub_prob and root in tritone_map:
                jazz_chords.append(f"{tritone_map[root]}7")
                continue
                
        # Add extensions
        if chord.endswith('7') and random.random() < 0.7:
            jazz_chords.append(chord.replace('7', '9'))
            continue
            
        jazz_chords.append(chord)
        
    return jazz_chords



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
        # Convert all chords to 7th chords
        chord_list = convert_to_seventh_chords(chord_list, SEVENTH_PROBABILITY)
        
        export_music(score, chord_list, gap_list, filename)
