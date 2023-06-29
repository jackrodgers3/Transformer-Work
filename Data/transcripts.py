

with open('inception.txt', 'r', encoding='utf-8') as f:
    text_lines = f.readlines()
f.close()
with open('inception.txt', 'r', encoding='utf-8') as f:
    text = f.read()
f.close()

vocab_size = sorted(list(set(text)))
print(f'Length of text: {len(text)}')
print(f'Vocab size of text: {len(vocab_size)}')
print('Vocab: ', ''.join(vocab_size))
print('')


def clean_data(corpus_lines):
    newline_counts = corpus_lines.count('\n')
    for i in range(len(corpus_lines) - newline_counts):
        if corpus_lines[i] == '\n':
            corpus_lines.pop(i)
    for j in range(len(corpus_lines)):
        ind1 = corpus_lines[j].find('\n')
        if ind1 != -1:
            corpus_lines[j] = corpus_lines[j][:ind1]
    for k in range(len(corpus_lines)):
        ind2 = corpus_lines[k].find('?')
        if ind2 != -1:
            corpus_lines[k] = corpus_lines[k][:ind2]
    for l in range(len(corpus_lines)):
        ind3 = corpus_lines[l].find('.')
        if ind3 != -1:
            corpus_lines[l] = corpus_lines[l].replace('.', '')
    for m in range(len(corpus_lines)):
        ind4 = corpus_lines[m].find(',')
        if ind4 != -1:
            corpus_lines[m] = corpus_lines[m].replace(',', '')
    for m1 in range(len(corpus_lines)):
        ind4_1 = corpus_lines[m1].find('"')
        if ind4_1 != -1:
            corpus_lines[m1] = corpus_lines[m1].replace('"', '')
    for m2 in range(len(corpus_lines)):
        ind4_2 = corpus_lines[m2].find('$')
        if ind4_2 != -1:
            corpus_lines[m2] = corpus_lines[m2].replace('$', '')
    for n in range(len(corpus_lines)):
        ind5 = corpus_lines[n].find(':')
        if ind5 != -1:
            corpus_lines[n] = corpus_lines[n].replace(':', '')
    for o in range(len(corpus_lines)):
        corpus_lines[o] = corpus_lines[o].lower()
    for p in range(len(corpus_lines)):
        ind6 = corpus_lines[p].find('!')
        if ind6 != -1:
            corpus_lines[p] = corpus_lines[p].replace('!', '')
    for q in range(len(corpus_lines)):
        ind7 = corpus_lines[q].find('i’ve')
        if ind7 != -1:
            corpus_lines[q] = corpus_lines[q].replace('i’ve', 'i have')
    for r in range(len(corpus_lines)):
        ind8 = corpus_lines[r].find('i’ll')
        if ind8 != -1:
            corpus_lines[r] = corpus_lines[r].replace('i’ll', 'i will')
    for s in range(len(corpus_lines)):
        ind9 = corpus_lines[s].find('it’s')
        if ind9 != -1:
            corpus_lines[s] = corpus_lines[s].replace('it’s', 'it is')
    for t in range(len(corpus_lines)):
        ind10 = corpus_lines[t].find('he’s')
        if ind10 != -1:
            corpus_lines[t] = corpus_lines[t].replace('he’s', 'he is')
    for u in range(len(corpus_lines)):
        ind11 = corpus_lines[u].find('you’re')
        if ind11 != -1:
            corpus_lines[u] = corpus_lines[u].replace('you’re', 'you are')
    for v in range(len(corpus_lines)):
        ind12 = corpus_lines[v].find('we’re')
        if ind12 != -1:
            corpus_lines[v] = corpus_lines[v].replace('we’re', 'we are')
    for w in range(len(corpus_lines)):
        ind13 = corpus_lines[w].find('he’ll')
        if ind13 != -1:
            corpus_lines[w] = corpus_lines[w].replace('he’ll', 'he will')
    for x in range(len(corpus_lines)):
        ind14 = corpus_lines[x].find('that’ll')
        if ind14 != -1:
            corpus_lines[x] = corpus_lines[x].replace('that’ll', 'that will')
    return corpus_lines
def get_unique_words(corpus_lines, num_lines):
    unique_words = []
    corpus_lines = clean_data(corpus_lines)
    for i in range(num_lines):
        words_in_line = corpus_lines[i].split(' ')
        for j in range(len(words_in_line)):
            if unique_words.count(words_in_line[j]) == 0:
                unique_words.append(words_in_line[j])
    #unique_words.append('!')
    #unique_words.append('?')
    #unique_words.append(':')
    #unique_words.append(',')
    #unique_words.append('.')
    unique_words.append(' ')
    unique_words.append('\n')
    print(f'Number of unique words in {num_lines} lines: {len(unique_words)}')
    return sorted(unique_words)

unique_words = get_unique_words(text_lines, 1690)
text = text.lower().replace('.', '').replace('!', '').replace('?', '').replace(':', '').replace(',','')\
    .replace('i’ve', 'i have').replace('i’ll', 'i will').replace('it’s', 'it is').replace('he’s', 'he is')\
    .replace('you’re', 'you are').replace('we’re', 'we are').replace('he’ll', 'he will')\
    .replace('that’ll', 'that will')

text1 = text.split('\n\n')
text3 = []
for i in range(1670):
    if text1[i].find(' ') != -1:
        text2 = text1[i].split(' ')
        for j in range(len(text2)):
            text3.append(text2[j])
    text3.append('\n')
    text3.append('\n')
#print(text3)

#making encoder (string -> #) and decoder (# -> string)
stoi = {ch: i for i, ch in enumerate(unique_words)}
itos = {i:ch for i,ch in enumerate(unique_words)}

encode = lambda line: [stoi[word] for word in text3[:line]]
decode = lambda l: ''.join([itos[i] for i in l])
print(stoi)
print(encode(6))