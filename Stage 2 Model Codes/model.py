import torch
import torch.nn as nn
from config import Config
import torch.nn.functional as F
import re

class HarmonicAttention(nn.Module):
    def __init__(self, chord_to_idx):
        super().__init__()
        self.function_map = self.create_function_map(chord_to_idx)
        self.transition_matrix = self.create_transition_matrix()
        
    
    def create_function_map(self, chord_to_idx):
        function_map = {}
        for chord in chord_to_idx:
            root = re.sub(r'[^A-G#b]', '', chord).upper()
            
            # Jazz-specific function mapping
            if 'dim' in chord:
                function_map[chord] = 'D'  # Diminished as dominant
            elif 'aug' in chord:
                function_map[chord] = 'T'  # Augmented as tonic
            elif any(ext in chord for ext in ['7', '9', '11', '13']):
                if 'm' in chord:
                    function_map[chord] = 'S'  # Minor 7/9/11 as subdominant
                elif 'sus' in chord:
                    function_map[chord] = 'D'  # Suspended as dominant
                else:
                    function_map[chord] = 'D'  # Dominant extensions
            elif '6' in chord or '69' in chord:
                function_map[chord] = 'T'  # 6th chords as tonic
            elif 'maj' in chord.lower():
                function_map[chord] = 'T'  # Major 7th as tonic
            else:
                # Fallback to jazz functional harmony
                function_map[chord] = 'T' if root in ['I'] else 'S' if root in ['II','IV'] else 'D'
        return function_map

    
    def create_transition_matrix(self):
        return {
            'T': {'S': 0.8, 'D': 0.9, 'T': 0.1},
            'S': {'D': 1.0, 'T': 0.7, 'S': 0.2},
            'D': {'T': 1.0, 'S': 0.3, 'D': 0.1}
        }
    
    def forward(self, current, next_chord):
        curr_func = self.function_map.get(current, 'T')
        next_func = self.function_map.get(next_chord, 'T')
        return torch.tensor(self.transition_matrix[curr_func][next_func], dtype=torch.float32)

class JazzHarmonizer(nn.Module):
    def __init__(self, input_dim, output_dim, chord_to_idx):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, Config.emb_dim)
        self.encoder = nn.LSTM(
            Config.emb_dim,
            Config.enc_hid_dim,
            bidirectional=True,
            batch_first=True,
            dropout=Config.dropout
        )
        self.attention = nn.Sequential(
            nn.Linear(Config.dec_hid_dim + 2*Config.enc_hid_dim, Config.attention_dim),
            nn.Tanh(),
            nn.Linear(Config.attention_dim, 1)
        )
        self.harmonic_attention = HarmonicAttention(chord_to_idx)
        self.hidden_proj = nn.Linear(2*Config.enc_hid_dim, Config.dec_hid_dim)
        self.cell_proj = nn.Linear(2*Config.enc_hid_dim, Config.dec_hid_dim)
        self.decoder = nn.LSTM(
            Config.emb_dim + 2*Config.enc_hid_dim,
            Config.dec_hid_dim,
            batch_first=True,
            dropout=Config.dropout
        )
        self.fc = nn.Linear(Config.dec_hid_dim, output_dim)

    def forward(self, src, trg):
        embedded = self.embedding(src)
        encoder_outputs, (hidden, cell) = self.encoder(embedded)
        hidden = self.hidden_proj(torch.cat((hidden[-2], hidden[-1]), dim=1)).unsqueeze(0)
        cell = self.cell_proj(torch.cat((cell[-2], cell[-1]), dim=1)).unsqueeze(0)
        
        batch_size, seq_len = trg.size(0), trg.size(1)
        outputs = []
        
        for t in range(seq_len):
            hidden_for_attn = hidden.permute(1, 0, 2)
            hidden_expanded = hidden_for_attn.expand(-1, encoder_outputs.size(1), -1)
            energy_input = torch.cat((hidden_expanded, encoder_outputs), dim=2)
            energy = self.attention(energy_input).squeeze(2)
            attn_weights = F.softmax(energy, dim=1).unsqueeze(1)
            context = torch.bmm(attn_weights, encoder_outputs)
            
            trg_emb = self.embedding(trg[:, t].unsqueeze(1))
            decoder_input = torch.cat([trg_emb, context], dim=2)
            
            output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            outputs.append(output)
            
        outputs = torch.cat(outputs, dim=1)
        return self.fc(outputs)
    
    
