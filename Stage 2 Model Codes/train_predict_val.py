import os
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from data_processing import create_dataset, load_preprocessed_data
from model import JazzHarmonizer
from config import Config
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import F1Score
import re
from sklearn.model_selection import train_test_split
import json
from datetime import datetime
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")



class MetricTracker:
    def __init__(self):
        self.train_loss = []
        self.train_acc = []
        self.train_f1 = []
        self.val_loss = []
        self.val_acc = []
        self.val_f1 = []
        self.best_metrics = {
            'loss': float('inf'),
            'acc': 0.0,
            'f1': 0.0,
            'epoch': -1
        }

    def update(self, epoch, t_loss, t_acc, t_f1, v_loss, v_acc, v_f1):
        self.train_loss.append(t_loss)
        self.train_acc.append(t_acc)
        self.train_f1.append(t_f1)
        self.val_loss.append(v_loss)
        self.val_acc.append(v_acc)
        self.val_f1.append(v_f1)
        
        # Update best metrics based on validation loss (can change to acc/f1 if preferred)
        if v_loss < self.best_metrics['loss']:
            self.best_metrics = {
                'loss': v_loss,
                'acc': v_acc,
                'f1': v_f1,
                'epoch': epoch+1
            }

    def save(self, path):
        metrics = {
            'train_loss': self.train_loss,
            'train_acc': self.train_acc,
            'train_f1': self.train_f1,
            'val_loss': self.val_loss,
            'val_acc': self.val_acc,
            'val_f1': self.val_f1,
            'best_metrics': self.best_metrics
        }
        with open(path, 'w') as f:
            json.dump(metrics, f)


class ChordDataset(Dataset):
    def __init__(self, df, chord_to_idx):
        self.data = []
        for _, row in df.iterrows():
            src = [chord_to_idx.get(c, 0) for c in row['normal'].split()[:Config.input_seq_length]]
            trg = [1] + [chord_to_idx[c] for c in row['jazz'].split()[:Config.output_seq_length]] + [0]
            self.data.append((src, trg))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return torch.LongTensor(self.data[idx][0]), torch.LongTensor(self.data[idx][1])

def collate_fn(batch):
    srcs, trgs = zip(*batch)
    max_src_len = max(len(s) for s in srcs)
    max_trg_len = max(len(t) for t in trgs)
    srcs_padded = [torch.cat((s, torch.zeros(max_src_len - len(s), dtype=torch.long))) for s in srcs]
    trgs_padded = [torch.cat((t, torch.zeros(max_trg_len - len(t), dtype=torch.long))) for t in trgs]
    return torch.stack(srcs_padded), torch.stack(trgs_padded)

def split_df(df, val_ratio=0.1, random_state=42):
    train_df, val_df = train_test_split(df, test_size=val_ratio, random_state=random_state, shuffle=True)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

def evaluate(model, val_loader, criterion, f1_metric):
    model.eval()
    total_loss = 0
    total_acc = 0
    total_f1 = 0
    total_tokens = 0  # Track actual non-padded tokens
    
    with torch.no_grad():
        for src, trg in val_loader:
            src = src.to(device)
            trg = trg.to(device)
            output = model(src, trg[:, :-1])
            
            # Flatten and mask like training
            output_flat = output.contiguous().view(-1, output.shape[-1])
            targets_flat = trg[:, 1:].contiguous().view(-1)
            mask = (targets_flat != 0)
            
            # Calculate loss only on relevant tokens
            loss = criterion(output_flat[mask], targets_flat[mask])
            
            # Get predictions and metrics
            preds = output.argmax(-1)
            batch_mask = (trg[:, 1:] != 0)
            correct = (preds == trg[:, 1:]).masked_select(batch_mask)
            
            # Accumulate properly weighted by tokens
            total_loss += loss.item() * mask.sum().item()
            total_acc += correct.sum().item()
            total_f1 += f1_metric(
                preds.masked_select(batch_mask),
                trg[:, 1:].masked_select(batch_mask)
            ).item() * batch_mask.sum().item()
            total_tokens += batch_mask.sum().item()

    model.train()
    # Avoid division by zero
    if total_tokens == 0:
        return 0.0, 0.0, 0.0
        
    return (
        total_loss / total_tokens,  # Per-token loss
        total_acc / total_tokens,   # Accuracy
        total_f1 / total_tokens     # F1 score
    )



def train():
    if os.path.exists(Config.save_path):
        print(f"Loading preprocessed data from {Config.save_path}")
        df, chord_to_idx, idx_to_chord = load_preprocessed_data(Config.save_path)
    else:
        print("Creating new dataset:")
        df, chord_to_idx, idx_to_chord = create_dataset(Config.corpus_dir, Config.save_path)

    # Split into train/validation sets
    train_df, val_df = split_df(df, val_ratio=0.1)
    train_loader = DataLoader(
        ChordDataset(train_df, chord_to_idx),
        batch_size=Config.batch_size,
        collate_fn=collate_fn,
        shuffle=True
    )
    val_loader = DataLoader(
        ChordDataset(val_df, chord_to_idx),
        batch_size=Config.batch_size,
        collate_fn=collate_fn,
        shuffle=False
    )

    model = JazzHarmonizer(
        len(chord_to_idx)+2,
        len(chord_to_idx)+2,
        chord_to_idx
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr)
    all_chords = list(chord_to_idx.keys())
    num_classes = len(chord_to_idx) + 2
    chord_counts = df['jazz'].str.split(expand=True).stack().value_counts()
    weights_series = (1.0 / np.sqrt(chord_counts + 1e-6)).reindex(all_chords, fill_value=1.0)
    weights = torch.ones(num_classes, dtype=torch.float32).to(device)
    for idx, chord in enumerate(all_chords, start=2):
        weights[idx] = weights_series.get(chord, 1.0)
    criterion = nn.CrossEntropyLoss(
        weight=weights,
        ignore_index=0
    )
    sample_input = "C G Am F C G Am F"
    best_loss = float('inf')

    f1_metric = F1Score(
        task='multiclass',
        num_classes=len(chord_to_idx)+2,
        ignore_index=0,
        average='macro'
    ).to(device)

    metrics = MetricTracker()

    checkpoint_files = [f for f in os.listdir('.') if f.startswith('model_epoch_') and f.endswith('.pth')]
    if checkpoint_files:
        latest_ckpt = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        print(f"Resuming from checkpoint: {latest_ckpt}")
        checkpoint = torch.load(latest_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('loss', best_loss)
        if 'metrics' in checkpoint:
            metrics = checkpoint['metrics']
    else:
        print("No checkpoint found, starting from scratch.")
        start_epoch = 0

    for epoch in range(start_epoch, Config.epochs):
        model.train()
        epoch_loss = 0
        epoch_acc = 0
        epoch_f1 = 0
        total_tokens = 0
        with tqdm(train_loader, unit="batch", desc=f"Epoch {epoch+1}/{Config.epochs}") as pbar:
            for src, trg in pbar:
                src = src.to(device)
                trg = trg.to(device)
                optimizer.zero_grad()
                output = model(src, trg[:, :-1])
                output_dim = output.shape[-1]
                loss = criterion(
                    output.contiguous().view(-1, output_dim),
                    trg[:, 1:].contiguous().view(-1)
                )
                preds = output.argmax(-1)
                mask = (trg[:, 1:] != 0)
                correct = (preds == trg[:, 1:]).masked_select(mask)
                acc = correct.sum().float() / mask.sum()
                f1 = f1_metric(
                    preds.masked_select(mask),
                    trg[:, 1:].masked_select(mask)
                )
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * mask.sum().item()
                epoch_acc += acc.item()
                epoch_f1 += f1.item()
                total_tokens += mask.sum().item()
                pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc.item():.4f}", f1=f"{f1.item():.4f}")

        # Validation step
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, f1_metric)
        print(f"Epoch {epoch+1} | Train Loss: {epoch_loss/len(train_loader):.4f} | Train Acc: {epoch_acc/len(train_loader):.4f} | Train F1: {epoch_f1/len(train_loader):.4f}")
        print(f"Epoch {epoch+1} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        # After validation step
        avg_train_loss = epoch_loss / total_tokens if total_tokens > 0 else 0
        avg_train_acc = epoch_acc / total_tokens
        avg_train_f1 = epoch_f1 / total_tokens
        
        metrics.update(epoch, epoch_loss/len(train_loader), epoch_acc/len(train_loader), epoch_f1/len(train_loader),
                      val_loss, val_acc, val_f1)
        

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f'model_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss/len(train_loader),
                'accuracy': epoch_acc/len(train_loader),
                'metrics': metrics
            }, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

        # Generate sample prediction
        model.eval()
        with torch.no_grad():
            prediction = predict(sample_input, model, chord_to_idx, idx_to_chord)
        model.train()
        print(f"Sample Prediction: '{sample_input}' → '{prediction}'\n")

        # Save best model based on validation loss
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_f1': val_f1,
                'metrics': metrics
            }, 'best_model.pth')

    metrics.save(f'metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    print(f"\nBest model at epoch {metrics.best_metrics['epoch']}:")
    print(f"Val Loss: {metrics.best_metrics['loss']:.4f}")
    print(f"Val Acc: {metrics.best_metrics['acc']:.4f}")
    print(f"Val F1: {metrics.best_metrics['f1']:.4f}")
    
    return model, chord_to_idx, idx_to_chord

def beam_search(model, input_ids, chord_to_idx, idx_to_chord, beam_width=3):
    model.eval()
    with torch.no_grad():
        input_tensor = torch.LongTensor([input_ids]).to(device)
        embedded = model.embedding(input_tensor)
        encoder_outputs, (hidden, cell) = model.encoder(embedded)
        hidden = model.hidden_proj(torch.cat((hidden[-2], hidden[-1]), dim=1)).unsqueeze(0)
        cell = model.cell_proj(torch.cat((cell[-2], cell[-1]), dim=1)).unsqueeze(0)
        beams = [{
            'score': 0.0,
            'seq': [1], # Start with SOS token
            'hidden': hidden,
            'cell': cell,
            'last_chord': None,
            'prev_chords': []
        }]
        for _ in range(len(input_ids)):
            new_beams = []
            for beam in beams:
                if beam['last_chord'] == 0:
                    new_beams.append(beam)
                    continue
                decoder_input = torch.LongTensor([[beam['seq'][-1]]]).to(device)
                decoder_emb = model.embedding(decoder_input)
                hidden_for_attn = beam['hidden'].permute(1,0,2)
                energy_input = torch.cat((
                    hidden_for_attn.expand(-1, encoder_outputs.size(1), -1),
                    encoder_outputs
                ), dim=2)
                attn_weights = F.softmax(model.attention(energy_input).squeeze(2), dim=1)
                context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
                decoder_in = torch.cat([decoder_emb, context], dim=2)
                output, (new_h, new_c) = model.decoder(decoder_in, (beam['hidden'], beam['cell']))
                log_probs = F.log_softmax(model.fc(output), dim=-1)
                top_probs, top_indices = log_probs.topk(beam_width)
                for i in range(beam_width):
                    token = top_indices[0,0,i].item()
                    new_score = beam['score'] + top_probs[0,0,i].item()
                    current_chord = idx_to_chord.get(beam['last_chord'], '') if beam['last_chord'] else None
                    next_chord = idx_to_chord.get(token, '')
                    current_beam_chords = beam['prev_chords'].copy()
                    if len(current_beam_chords) > 0:
                        last_chord = current_beam_chords[-1]
                        if next_chord == last_chord:
                            new_score /= Config.repeat_penalty
                    if current_chord and next_chord:
                        harmonic_score = model.harmonic_attention(current_chord, next_chord)
                        new_score += harmonic_score.item() * Config.harmonic_weight
                    if any(ext in next_chord for ext in ['7', '9', '11', '13', 'add']):
                        new_score += Config.extension_bonus
                    if 'dim' in next_chord or 'aug' in next_chord or '+' in next_chord or 'o' in next_chord:
                        new_score += Config.dim_aug_bonus
                    if current_chord and next_chord:
                        current_root = re.sub(r'[^A-G#b]', '', current_chord).upper()
                        next_root = re.sub(r'[^A-G#b]', '', next_chord).upper()
                        if (current_root, next_root) in [('II', 'V'), ('V', 'I'), ('VI', 'II')]:
                            new_score += Config.progression_bonus
                    new_beams.append({
                        'score': new_score,
                        'seq': beam['seq'] + [token],
                        'hidden': new_h,
                        'cell': new_c,
                        'last_chord': token,
                        'prev_chords': current_beam_chords + [next_chord]
                    })
            beams = sorted(new_beams, key=lambda x: x['score'], reverse=True)[:beam_width]
        best_seq = beams[0]['seq']
        return ' '.join([idx_to_chord.get(idx, '') for idx in best_seq if idx not in [0,1]])

def predict(progression, model, chord_to_idx, idx_to_chord):
    return beam_search(model,
        [chord_to_idx.get(c, 0) for c in progression.split()],
        chord_to_idx,
        idx_to_chord
    )

if __name__ == "__main__":
    trained_model, chord_to_idx, idx_to_chord = train()
    
    # Common progressions from search results + jazz standards
    test_progressions = [
        # Basic progressions
        "C F G",                  # I-IV-V
        "C G Am F",               # I-V-vi-IV (Pop)
        "C Am F G",               # vi-IV-I-V (Jazz variation)
        "C Dm G",                 # ii-V-I (Jazz)
        "C E Am F",               # I-III-vi-IV (Bluesy)
        "C Fm C G",               # I-iv-I-V (Minor blend)
        "C Bdim Em Am",           # vii°-iii-vi (Ragtime)
        
        # Extended progressions
        "C C7 F Fm C G7 C",       # Blues turnaround
        "Dm7 G7 Cmaj7",           # ii-V-I jazz
        "Cmaj7 Am7 Dm7 G7",       # Jazz quatrain
        "Fmaj7 Bb7 Eb7 Am7 D7 Gm7 C7",  # Coltrane changes
        
        # Minor key tests
        "Am G F E",               # Andalusian cadence
        "Am Dm E7 Am",            # Minor ii-V-i
        "Am E7 Am Dm G7 C F E7"   # Flamenco/jazz fusion
    ]

    # print("\n" + "="*50)
    # print("Jazz Harmonization Predictions")
    # print("="*50)
    
    # for progression in test_progressions:
    #     try:
    #         prediction = predict(
    #             progression, 
    #             trained_model,
    #             chord_to_idx,
    #             idx_to_chord
    #         )
    #         print(f"\nInput: {progression}")
    #         print(f"Output: {prediction}")
    #         print("-"*50)
    #     except Exception as e:
    #         print(f"Error processing '{progression}': {str(e)}")


if __name__ == "__main__":
    trained_model, chord_to_idx, idx_to_chord = train()

    # Basic progressions in all major sharp and flat keys, lengths 3 to 16
    test_progressions = [
        # Short (3-4 chords)
        "C F G",           # C major
        "G C D",           # G major (1 sharp)
        "D G A",           # D major (2 sharps)
        "A D E",           # A major (3 sharps)
        "E A B",           # E major (4 sharps)
        "B E F#",          # B major (5 sharps)
        "F# B C#",         # F# major (6 sharps)
        "F Bb C",          # F major (1 flat)
        "Bb Eb F",         # Bb major (2 flats)
        "Eb Ab Bb",        # Eb major (3 flats)
        "Ab Db Eb",        # Ab major (4 flats)
        "Db Gb Ab",        # Db major (5 flats)
        "Gb Cb Db",        # Gb major (6 flats)
        "Cb Fb Gb",        # Cb major (7 flats, enharmonic to B)

        # Medium (6-8 chords)
        "C G Am F C G Am F",           # C major
        "G D Em C G D Em C",           # G major
        "D A Bm G D A Bm G",           # D major
        "A E F#m D A E F#m D",         # A major
        "E B C#m A E B C#m A",         # E major
        "B F# G#m E B F# G#m E",       # B major
        "F# C# D#m B F# C# D#m B",     # F# major
        "F C Dm Bb F C Dm Bb",         # F major
        "Bb F Gm Eb Bb F Gm Eb",       # Bb major
        "Eb Bb Cm Ab Eb Bb Cm Ab",     # Eb major
        "Ab Eb Fm Db Ab Eb Fm Db",     # Ab major
        "Db Ab Bbm Gb Db Ab Bbm Gb",   # Db major
        "Gb Db Ebm Cb Gb Db Ebm Cb",   # Gb major
        "Cb Gb Abm Fb Cb Gb Abm Fb",   # Cb major

        # Long (12 chords)
        "C F G C F G C F G C F G",           # C major
        "G C D G C D G C D G C D",           # G major
        "D G A D G A D G A D G A",           # D major
        "F Bb C F Bb C F Bb C F Bb C",       # F major
        "Bb Eb F Bb Eb F Bb Eb F Bb Eb F",   # Bb major
        "E A B E A B E A B E A B",           # E major
        "B E F# B E F# B E F# B E F#",       # B major

        # Very long (16 chords)
        "C G Am F C G Am F C G Am F C G Am F",           # C major
        "G D Em C G D Em C G D Em C G D Em C",           # G major
        "D A Bm G D A Bm G D A Bm G D A Bm G",           # D major
        "A E F#m D A E F#m D A E F#m D A E F#m D",       # A major
        "E B C#m A E B C#m A E B C#m A E B C#m A",       # E major
        "B F# G#m E B F# G#m E B F# G#m E B F# G#m E",   # B major
        "F C Dm Bb F C Dm Bb F C Dm Bb F C Dm Bb",       # F major
        "Bb F Gm Eb Bb F Gm Eb Bb F Gm Eb Bb F Gm Eb",   # Bb major
        "Eb Bb Cm Ab Eb Bb Cm Ab Eb Bb Cm Ab Eb Bb Cm Ab", # Eb major
        "Ab Eb Fm Db Ab Eb Fm Db Ab Eb Fm Db Ab Eb Fm Db", # Ab major
        "Db Ab Bbm Gb Db Ab Bbm Gb Db Ab Bbm Gb Db Ab Bbm Gb", # Db major
        "Gb Db Ebm Cb Gb Db Ebm Cb Gb Db Ebm Cb Gb Db Ebm Cb", # Gb major
        "Cb Gb Abm Fb Cb Gb Abm Fb Cb Gb Abm Fb Cb Gb Abm Fb", # Cb major

        "C G Am F C G Am F"
    ]

    # print("\n" + "="*60)
    # print("Jazz Harmonization Predictions for Various Progressions")
    # print("="*60)

    # for progression in test_progressions:
    #     try:
    #         prediction = predict(
    #             progression,
    #             trained_model,
    #             chord_to_idx,
    #             idx_to_chord
    #         )
    #         print(f"\nInput:  {progression}")
    #         print(f"Output: {prediction}")
    #         print("-" * 60)
    #     except Exception as e:
    #         print(f"Error processing '{progression}': {str(e)}")


