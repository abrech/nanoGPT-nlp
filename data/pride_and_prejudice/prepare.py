"""
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import requests
import numpy as np
import argparse
# preprocessing stuff
import re
# BPE stuff
import itertools

# download pride and prejudice (totally legal)
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://www.gutenberg.org/cache/epub/1342/pg1342.txt'
    with open(input_file_path, 'w', encoding="utf-8") as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r', encoding="utf-8") as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

# Remove [Illustration: ...] using good old finger counting
# (regex with nested brackets is not ok)
look_for = '[Illustration'
def remove_illustrations(text):
    result = []
    i = 0
    while i < len(text):
        if text[i:i+len(look_for)] == look_for:
            depth = 0
            while i < len(text):
                if text[i] == '[':
                    depth += 1
                elif text[i] == ']':
                    depth -= 1
                    if depth == 0:
                        i += 1
                        break
                i += 1
        else:
            result.append(text[i])
            i += 1
    return ''.join(result)

data = remove_illustrations(data)

# remove underscores: they only emphasize words, we can omit that
data = data.replace("_", "")

# get all the unique characters that occur in this text
vocab = sorted(list(set(data)))
vocab_size = len(vocab)
print("all the unique characters:", ''.join(vocab))
print(f"vocab size: {vocab_size:,}")

# BPE starting here
parser = argparse.ArgumentParser()
parser.add_argument('--k', type=int, default=500, help='number of BPE merge operations')
args = parser.parse_args()
k = args.k


END_OF_WORD = " "
vocab = sorted(list(set(data)))
# we are not interested in special characters, as they are
# usually (hopefully) not part of the words and already in the vocab
words = [re.sub(r'[^a-zA-Z0-9]+$', '', word) for word in re.split('-- | ', data)]
corpus = [list(word) + [" "] for word in words if word]

for _ in range(k):
    print(f"BPE iteration {_ + 1}")
    occs = dict()
    for word in corpus:
        for i in range(len(word) - 1):
            merge = word[i] + word[i+1]
            occs.update({merge: occs.get(merge, 0) + 1})
    max_occ = max(occs, key=occs.get)
    vocab.append(max_occ)
    print(max_occ)

    for word in corpus:
        i = 0
        while i < len(word) - 1:
            merge = word[i] + word[i+1]
            if merge == max_occ:
                word[i] = merge
                word.pop(i+1)
            else:
                i += 1
print(vocab)

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(vocab) }
itos = { i:ch for i,ch in enumerate(vocab) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train and test splits
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

# length of dataset in characters:  1115394
# all the unique characters:
#  !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
# vocab size: 65
# train has 1003854 tokens
# val has 111540 tokens
