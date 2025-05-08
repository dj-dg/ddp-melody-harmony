from pathlib import Path
import pandas as pd
import re
import torch
from music21 import harmony
from tqdm import tqdm  # <-- Progress bar
from torch.serialization import safe_globals

corpus_dir = "/Users/dg/DDP/jazz_chord_progression/SongDB"          # Root directory containing .txt files
out_path = "/Users/dg/DDP/jazz_chord_progression/preprocessed_data.bin"  # Preprocessed data path

def extract_progressions(file_path):
    """Extract chord progressions from a corpus file"""
    progressions = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or '=' in line:
                continue
            chords = []
            for bar in line.split('|'):
                bar = bar.strip()
                if bar:
                    chords.extend(bar.split())
            if chords:
                progressions.append(chords)
    return progressions

def simplify_chord_new(jazz_chord):
    if len(jazz_chord) == 1:
        return jazz_chord
    if len(jazz_chord) == 2:
        if jazz_chord[-1] == 'b' or jazz_chord[-1] == 'm':
            return jazz_chord
        else:
            return jazz_chord[0]
    if len(jazz_chord) >= 3:
        jazz_chord_new = jazz_chord[0] + jazz_chord[1] + jazz_chord[2]
        if jazz_chord_new[-1] == 'm':
            return jazz_chord_new
        elif jazz_chord_new[1] == 'm' or jazz_chord_new[1] == '#' or jazz_chord_new[1] == 'b':
            return jazz_chord_new[0] + jazz_chord_new[1]
        else:
            return jazz_chord_new[0] 
        

def simplify_chord(jazz_chord):
    """Convert jazz chord to basic triad with quality, preserving accidentals and handling maj7 chords"""
    try:
        h = harmony.ChordSymbol(jazz_chord)
        root = h.root().name
        
        # Handle chord quality
        if h.quality == 'minor':
            quality = 'm'
        elif 'major' in h.chordKind:  # Handle maj7 chords specifically
            quality = ''
        else:
            quality = 'm' if h.quality == 'minor' else ''
            
        return f"{root}{quality}"
    except:
        # Improved regex that preserves accidentals and handles maj/M suffixes
        base = re.sub(
            r'(?i)(maj|M|7|9|11|13|6|#9|b5|/.*|add|sus|dim|aug)',
            '', 
            jazz_chord
        )
        # Clean up double accidentals and preserve case
        base = base.replace('##', '#').replace('bb', 'b')
        
        # Extract root with accidentals using positive lookahead
        root_match = re.match(r'^([A-Ga-g#b]+)', base)
        if root_match:
            cleaned_root = root_match.group(1).upper()
            # Preserve accidental case (B vs b)
            return f"{cleaned_root}{'m' if 'm' in jazz_chord.lower() else ''}"
        return base


# from music21 import converter, harmony, key, roman

# def progression_to_degrees(progression_str):
#     """Convert chord progression to Roman numerals with explicit natural minor handling"""
#     if not progression_str:
#         return ''
    
#     chords = progression_str.split()
#     if not chords:
#         return ''

#     # Detect key with natural minor enforcement
#     try:
#         s = converter.parse(' '.join(chords), format='romanText')
#         key_analysis = s.analyze('key')
#         if key_analysis.mode == 'minor':
#             music21_key = key.Key(key_analysis.tonic.name, 'natural')
#         else:
#             music21_key = key_analysis
#     except:
#         # Fallback: create key from first chord (natural minor if minor)
#         try:
#             first_chord = harmony.ChordSymbol(chords[0])
#             root = first_chord.root().name.replace('-', 'b')  # Ensure flat notation
#             if 'm' in chords[0].lower():
#                 music21_key = key.Key(root, 'natural')
#             else:
#                 music21_key = key.Key(root)
#         except:
#             return ' '.join(['?' for _ in chords])

#     # Convert chords with proper key context
#     degrees = []
#     for chord_str in chords:
#         try:
#             cs = harmony.ChordSymbol(chord_str)
#             rn = roman.romanNumeralFromChord(cs, music21_key)
            
#             # Handle secondary analysis if standard conversion fails
#             if '?' in rn.figure:
#                 secondary_rn = roman.RomanNumeral(cs, music21_key)
#                 degrees.append(secondary_rn.figure)
#             else:
#                 degrees.append(rn.figure)
#         except:
#             degrees.append('?')

#     return ' '.join(degrees)


    


def create_dataset(root_dir, save_path=None):
    """Process all .txt files in directory and subdirectories, with progress meter"""
    jazz_progressions = []
    file_paths = list(Path(root_dir).rglob('*.txt'))
    
    print(f"Found {len(file_paths)} files. Extracting progressions...")
    for file_path in tqdm(file_paths, desc="Files processed"):
        jazz_progressions.extend(extract_progressions(file_path))
    
    print(f"Extracted {len(jazz_progressions)} progressions. Simplifying chords...")
    data = []
    for progression in tqdm(jazz_progressions, desc="Progressions processed"):
        normal_chords = [simplify_chord_new(c) for c in progression]
        data.append((' '.join(normal_chords), ' '.join(progression)))

    df = pd.DataFrame(data, columns=['normal', 'jazz'])
    
    all_chords = set()
    for progression in df['jazz']:
        all_chords.update(progression.split())
    chord_to_idx = {c:i+2 for i,c in enumerate(sorted(all_chords))}  # 0:pad, 1:sos
    idx_to_chord = {v:k for k,v in chord_to_idx.items()}
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'df': df,
            'chord_to_idx': chord_to_idx,
            'idx_to_chord': idx_to_chord
        }, save_path)
        print(f"Dataset saved to {save_path}")
    
    return df, chord_to_idx, idx_to_chord

def load_preprocessed_data(load_path):
    """Load dataset while allowing pandas DataFrame"""
    with safe_globals([pd.DataFrame]):  # Add this context manager
        data = torch.load(load_path, weights_only=False)
    return data['df'], data['chord_to_idx'], data['idx_to_chord']

# import torch
# import pandas as pd

# # Path to your preprocessed data file (from config.py or your own path)
# path = "/Users/dg/DDP/jazz_chord_progression/preprocessed_data.bin"

# # Load the data
# data = torch.load(path, weights_only=False)


# # View the DataFrame of progressions
# df = data['df']
# print(df.head())

# # View the chord-to-index mapping
# chord_to_idx = data['chord_to_idx']
# print(list(chord_to_idx.items())[:10])  # Show first 10 mappings

# # View the index-to-chord mapping
# idx_to_chord = data['idx_to_chord']
# print(list(idx_to_chord.items())[:10])  # Show first 10 mappings



df, chord_to_idx, idx_to_chord = create_dataset(
    root_dir=corpus_dir,
    save_path=out_path
)

print(df.head(20))
print(f"Vocabulary size: {len(chord_to_idx)}")
