import itertools

# BPE LETS GOOOOO
# store merges we already "found"
been_found = set()

k = 10
END_OF_WORD = " "
data = "low low low low low lowest lowest newer newer newer newer newer newer wider wider wider new new"
vocab = sorted(list(set(data)))
words = data.split(" ")
corpus = [list(word) + [" "] for word in words]
for c in corpus:
    print(c)
for _ in range(k):
    occs = dict()
    for word in corpus:
        for i in range(len(word) - 1):
            merge = word[i] + word[i+1]
            occs.update({merge: occs.get(merge, 0) + 1})
    max_occ = max(occs, key=occs.get)
    print(occs)
    print(max_occ)
    vocab.append(max_occ)

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