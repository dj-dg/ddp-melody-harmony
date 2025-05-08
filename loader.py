import os
import pickle
import numpy as np
from copy import deepcopy
from tqdm import trange
from music21 import *
from config import *

def ks2gap(ks):
    if isinstance(ks, key.KeySignature):
        ks = ks.asKey()
    try:
        # Use relative major for minor keys
        if ks.mode == 'major':
            tonic = ks.tonic
        else:
            tonic = ks.relative.major.tonic  # Changed from parallel.tonic
        gap = interval.Interval(tonic, pitch.Pitch('C'))
    except:
        gap = interval.Interval(0)
    return gap


def get_filenames(input_dir):
    filenames = []
    filecount = 0
    for dirpath, dirlist, filelist in os.walk(input_dir):
        for this_file in filelist:
            filename = os.path.join(dirpath, this_file)
            filenames.append(filename)
            filecount += 1
    print('Number of files:', filecount)
    return filenames

def harmony2idx(element):
    pitch_list = [sub_ele.pitch.midi for sub_ele in element.notes]
    pitch_list = sorted(pitch_list)
    bass_note = pitch_list[0]%12
    quality = pitch_list[min(1,len(pitch_list)-1)]-pitch_list[0]
    if quality<=3:
        quality = 0
    else:
        quality = 1
    return bass_note*2+quality

def melody_reader(score):
    melody_vecs = []
    chord_list = []
    gap_list = []
    last_chord = 0
    last_ks = key.KeySignature(0)
    for m in score.recurse():
        if isinstance(m, stream.Measure):
            measure_duration = m.duration.quarterLength
            half_duration = measure_duration / 2.0
            first_half_vec = [0.0] * 12
            second_half_vec = [0.0] * 12
            this_chord_first = None
            this_chord_second = None

            if m.keySignature is not None:
                gap = ks2gap(m.keySignature)
                last_ks = m.keySignature
            else:
                gap = ks2gap(last_ks)
            gap_list.extend([gap, gap])

            for n in m:
                element_start = n.offset
                element_duration = n.quarterLength
                element_end = element_start + element_duration

                def add_to_vec(element, duration, vec, gap):
                    if isinstance(element, note.Note):
                        pitch_idx = element.transpose(gap).pitch.midi % 12
                        vec[pitch_idx] += duration
                    elif isinstance(element, chord.Chord) and not isinstance(element, harmony.ChordSymbol):
                        pitches = [nn.transpose(gap).pitch.midi for nn in element.notes]
                        pitch_idx = sorted(pitches)[-1] % 12
                        vec[pitch_idx] += duration

                if element_end <= half_duration:
                    add_to_vec(n, element_duration, first_half_vec, gap)
                elif element_start >= half_duration:
                    add_to_vec(n, element_duration, second_half_vec, gap)
                else:
                    first_part = half_duration - element_start
                    second_part = element_duration - first_part
                    add_to_vec(n, first_part, first_half_vec, gap)
                    add_to_vec(n, second_part, second_half_vec, gap)

                if isinstance(n, harmony.ChordSymbol):
                    if n.offset < half_duration and this_chord_first is None:
                        this_chord_first = harmony2idx(n.transpose(gap)) + 1
                        last_chord = this_chord_first
                    elif n.offset >= half_duration:
                        this_chord_second = harmony2idx(n.transpose(gap)) + 1
                        last_chord = this_chord_second

            for idx, vec in enumerate([first_half_vec, second_half_vec]):
                total = sum(vec)
                if total > 0:
                    vec = np.array(vec, dtype=np.float32) / total
                else:
                    vec = np.array(vec, dtype=np.float32)
                melody_vecs.append(vec.tolist())
                if idx == 0:
                    chord_list.append(this_chord_first if this_chord_first is not None else last_chord)
                else:
                    chord_list.append(this_chord_second if this_chord_second is not None else last_chord)
                last_chord = chord_list[-1]

    return melody_vecs, chord_list, gap_list

def convert_files(filenames, fromDataset=True):
    print('\nConverting %d files...' %(len(filenames)))
    failed_list = []
    data_corpus = []
    for filename_idx in trange(len(filenames)):
        filename = filenames[filename_idx]
        try:
            score = converter.parse(filename)
            score = score.parts[0]
            if not fromDataset:
                original_score = deepcopy(score)
            song_data = []
            melody_vecs, chord_txt, gap_list = melody_reader(score)
            if fromDataset:
                song_data.append((melody_vecs, chord_txt))
            else:
                data_corpus.append((melody_vecs, gap_list, original_score, filename))
            if len(song_data)>0:
                data_corpus.append(song_data)
        except Exception as e:
            failed_list.append((filename, e))
    print('Successfully converted %d files.' %(len(filenames)-len(failed_list)))
    if len(failed_list)>0:
        print('Failed numbers: '+str(len(failed_list)))
        print('Failed to process: \n')
        for failed_file in failed_list:
            print(failed_file)
    if fromDataset:
        with open(CORPUS_PATH, "wb") as filepath:
            pickle.dump(data_corpus, filepath)
    else:
        return data_corpus

if __name__ == '__main__':
    filenames = get_filenames(input_dir=DATASET_PATH)
    convert_files(filenames)
