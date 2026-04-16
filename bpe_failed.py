import itertools

# BPE LETS GOOOOO
# store merges we already "found"
been_found = set()

data = "low low low low low lowest lowest newer newer newer newer newer newer wider wider wider new new"
vocab = sorted(list(set(data)))


for i in range(10):
    # isolate what we will look for
    pairs_raw = set(itertools.product(vocab, vocab))
    pairs = {"".join(p) for p in pairs_raw if not " " in p[0] and not " " in p[1] or p[1].endswith(" ")}
    # remove what has been found
    pairs -= been_found
    
    print(len(pairs))
    # count
    occ = {pair: data.count(pair) for pair in pairs}
    # most_occ_pair = max(occ, key=occ.get)
    # find max with length priority
    curr_max = -1
    curr_len = -1
    for k, v in occ.items():
        if v > curr_max or (v == curr_max and (len(k) > curr_len 
        or (len(k) == curr_len and not k.endswith(" ")))):
            curr_max = v
            curr_len = len(k)
            most_occ_pair = k

    been_found.add(most_occ_pair)
    vocab.append(most_occ_pair)
    print(most_occ_pair)

print(vocab)
# print(been_found)
print(pairs)