# Hyperparameters and paths
class Config:
    # Data parameters
    corpus_dir = "/Users/dg/DDP/jazz_chord_progression_v3/SongDB"          # Root directory containing .txt files
    save_path = "/Users/dg/DDP/jazz_chord_progression_v3/preprocessed_data.bin"  # Preprocessed data path
    input_seq_length = None
    output_seq_length = None
    batch_size = 64
    
    # Model architecture
    emb_dim = 128
    enc_hid_dim = 256
    dec_hid_dim = 512
    attention_dim = 200
    dropout = 0.3
    
    # Training
    epochs = 200
    lr = 0.001
    early_stop_patience = 20

    # ext_boost_factors = {
    #     '7': 1.2, '9': 1.3, '11': 1.4, '13': 1.5,
    #     '#5': 1.1, 'b5': 1.1, '#9': 1.2, 'b9': 1.2,
    #     'dim': 1.4, 'aug': 1.3, 'sus': 1.2, 'o': 1.4, '+': 1.3, 'add': 1.2
    # }


    harmonic_weight = 0.7
    # voice_leading_weight = 0.4
    beam_width = 3

    repeat_penalty = 1.5  # Penalty for consecutive repeats
    verbose_beam = False  # Set True to see penalty applications

    extension_bonus = 0.25  # Bonus for extended chords
    dim_aug_bonus = 0.15    # Bonus for diminished/augmented
    progression_bonus = 0.2 # Bonus for jazz progressions

    # teacher_forcing_start = 1.0  # Start with 100% teacher forcing
    # teacher_forcing_end = 0.1    # End with 10% teacher forcing
    # teacher_forcing_decay = 'linear'
