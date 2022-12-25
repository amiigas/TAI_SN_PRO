import pandas as pd
import numpy as np
from tqdm import tqdm
import os


train_df = pd.read_csv("./data/updated_train.csv")

D = set()
for seq in train_df.protein_sequence:
    D.update(list(seq))
D = sorted(D)

mapping = {char: val+1 for val, char in enumerate(D)}
protein_sequences = np.zeros((len(train_df), train_df.protein_sequence.str.len().max()), dtype=np.int8)

for i, seq in enumerate(tqdm(train_df.protein_sequence)):
    for j, char in enumerate(seq):
        protein_sequences[i][j] = mapping[char]

os.makedirs("./data/int_mapped_zero_padded")
np.save("./data/int_mapped_zero_padded/inputs.npy", protein_sequences)
np.save("./data/int_mapped_zero_padded/targets.npy", train_df.tm)

one_hot = np.eye(len(D))
one_hot_encoded_mapping = {char: one_hot[val] for val, char in enumerate(D)}
max_seq_len = train_df.protein_sequence.str.len().max()
protein_sequences = np.zeros((len(train_df), max_seq_len, len(D)), dtype=np.int8)

for i, seq in enumerate(tqdm(train_df.protein_sequence)):
    for j, char in enumerate(seq[::-1]):
        protein_sequences[i][max_seq_len-1-j] = one_hot_encoded_mapping[char]

os.makedirs("./data/one_hot_zero_padded")
np.save("./data/one_hot_zero_padded/inputs.npy", protein_sequences)
np.save("./data/one_hot_zero_padded/targets.npy", train_df.tm)
