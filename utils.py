import os
import numpy as np
import pandas as pd
FILEDIR = r'C:\Users\jackm\PycharmProjects\Transformer-Work\Data\BookCorpus\epubtxt/'
list = os.listdir(FILEDIR)
FILE_LIST = list[121:123]
EX_FILTERS = ['\n', ' ', '.']
EX_TEXT = "Hi. My name is Cobb and I am doing well. I am doing stuff. \nI hope one day, I can leave this place. \nIt is not fun."

def separator(text):
    text = text.lower()
    indd = -2
    indd2 = -2
    indd3 = -2
    indd4 = -2
    indd5 = -2
    while indd != -1:
        indd = text.find('_')
        if indd != -1:
            text = text[indd:] + text[:indd+1]
    while indd2 != -1:
        indd2 = text.find('\\')
        if indd2 != -1:
            text = text[indd2:] + text[:indd2+1]
    while indd3 != -1:
        indd3 = text.find('"')
        if indd3 != -1:
            text = text[indd3:] + text[:indd3+1]
    while indd4 != -1:
        indd4 = text.find('(')
        if indd4 != -1:
            text = text[indd4:] + text[:indd4+1]
    breakup = text.split(' ')
    breakup2 = []
    for i in range(len(breakup)):
        if i != len(breakup)-1:
            breakup2.append(breakup[i])
            breakup2.append(' ')
        else: breakup2.append(breakup[i])
    breakup3 = []
    for i in range(len(breakup2)):
        ind1 = breakup2[i].find('.')
        if ind1 != -1:
            breakup3.append(breakup2[i][:ind1])
            breakup3.append('.')
        else:
            breakup3.append(breakup2[i])
    breakup4 = []
    for i in range(len(breakup3)):
        ind2 = breakup3[i].find('\n')
        if ind2 != -1:
            breakup4.append('\n')
            breakup4.append('\n')
            breakup4.append(breakup3[i][ind2+2:])
        else:
            breakup4.append(breakup3[i])
    breakup5 = []
    for i in range(len(breakup4)):
        ind3 = breakup4[i].find(',')
        if ind3 != -1:
            breakup5.append(breakup4[i][:ind3])
            breakup5.append(',')
        else:
            breakup5.append(breakup4[i])
    breakup6 = []
    for i in range(len(breakup5)):
        ind4 = breakup5[i].find('?')
        if ind4 != -1:
            breakup6.append(breakup5[i][:ind4])
            breakup6.append('?')
        else:
            breakup6.append(breakup5[i])

    return breakup6

def simple_separator(text):
    #separates words, space, period, newline, question mark, and comma
    text = text.replace('"','')
    breakup = text.split(' ')
    breakup2 = []
    for i in range(len(breakup)):
        if i != len(breakup) - 1:
            breakup2.append(breakup[i])
            breakup2.append(' ')
        else:
            breakup2.append(breakup[i])
    breakup3 = []
    for i in range(len(breakup2)):
        ind1 = breakup2[i].find('.')
        if ind1 != -1:
            breakup3.append(breakup2[i][:ind1])
            breakup3.append('.')
            breakup3.append(breakup2[i][ind1+1:])
        else:
            breakup3.append(breakup2[i])
    ind2 = -2
    while ind2 != -1:
        try:
            ind2 = breakup3.index('')
            breakup3.pop(ind2)
        except ValueError:
            ind2 = -1
    breakup4 = []
    for i in range(len(breakup3)):
        ind3 = breakup3[i].find('\n')
        if ind3 != -1:
            breakup4.append(breakup3[i][:ind3])
            breakup4.append('\n')
            breakup4.append(breakup3[i][ind3+1:])
        else:
            breakup4.append(breakup3[i])
    ind4 = -2
    while ind4 != -1:
        try:
            ind4 = breakup4.index('')
            breakup4.pop(ind4)
        except ValueError:
            ind4 = -1
    breakup5 = []
    for i in range(len(breakup4)):
        ind5 = breakup4[i].find('?')
        if ind5 != -1:
            breakup5.append(breakup4[i][:ind5])
            breakup5.append('?')
        else:
            breakup5.append(breakup4[i])
    breakup6 = []
    for i in range(len(breakup5)):
        ind6 = breakup5[i].find(',')
        if ind6 != -1:
            breakup6.append(breakup5[i][:ind6])
            breakup6.append(',')
        else:
            breakup6.append(breakup5[i])
    return breakup6

def get_unique_tokens(filedir, filelist):
    ult_list = []
    total_text = ""
    filecounter = 0
    filtered_tokens = ['.', ',', '"', '?', '!', '\n', ' ', '_', '#', '(', ')', '*', '|']
    for b in range(len(filelist)):
        filecounter += 1
        with open(filedir+filelist[b], 'r', encoding='utf-8') as f:
            temp_text = f.read()
        f.close()
        total_text += temp_text
        text_list = temp_text.split('\n\n')

        for i in range(len(text_list)):
            sublist = text_list[i].lower().split(' ')
            for j in range(len(sublist)):
                #FILTERING CLUTTER
                for k in range(len(filtered_tokens)):
                    if sublist[j].find(filtered_tokens[k]) != -1:
                        sublist[j] = sublist[j].replace(filtered_tokens[k], '')
                #ADD TO MAIN UNIQUE WORD LIST
                try:
                    ind3 = ult_list.index(sublist[j])
                except ValueError:
                    ult_list.append(sublist[j])

    for c in range(len(filtered_tokens)):
        ult_list.append(filtered_tokens[c])
    print(f'# of unique tokens in {filecounter} text files: {len(ult_list)}')
    return sorted(ult_list), total_text




def maps(token_list):
    stoi = {ch: i for i, ch in enumerate(token_list)}
    itos = {i: ch for i, ch in enumerate(token_list)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    return encode, decode

if __name__ == '__main__':
    TOKEN_LIST, CORPUS = get_unique_tokens(FILEDIR, ['blood-moon.epub.txt'])
    print(TOKEN_LIST)
    text = separator(CORPUS)[:5000]
    print(text)
    ENCODER, DECODER = maps(token_list=TOKEN_LIST)
    print(ENCODER(text))
