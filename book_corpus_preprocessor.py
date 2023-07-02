import os

#book corpus preprocessor
#@author: Jack P. Rodgers

base_dir = r"C:\Users\jackm\PycharmProjects\Transformer-Work\Data\BookCorpus\epubtxt"

file_list = os.listdir(base_dir)

work_corpus = ""

n = 5

for i in range(n):
    with open(base_dir + '/' + file_list[i], 'r', encoding='utf-8') as f:
        work_corpus += f.read()
        f.close()

chars = sorted(list(set(work_corpus)))
print(f'Number of unique characters in first {n} texts: {len(chars)}')
chars1 = ''.join(chars)
print(f'Unique characters in first {n} texts: {chars1}')
print(f'Length of first {n} texts: {len(work_corpus)}')

with open(r"C:\Users\jackm\PycharmProjects\Transformer-Work\Data\work_corpus.txt", 'w', encoding='utf-8') as g:
    g.write(work_corpus)
    g.close()
