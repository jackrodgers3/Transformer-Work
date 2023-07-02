import string

import mido
import os

file_list = os.listdir(r'C:\Users\jackm\PycharmProjects\Transformer-Work\Data\Beethoven_MIDI')

mid_list = [mido.MidiFile('Data/Beethoven_MIDI/'+ file_list[i], clip=True) for i in range(len(file_list))]


def msg2dict(msg):
    result = dict()
    if 'note_on' in msg:
        on_ = True
    elif 'note_off' in msg:
        on_ = False
    else:
        on_ = None
    result['time'] = int(msg[msg.rfind('time'):].split(' ')[0].split('=')[1].translate(
        str.maketrans({a: None for a in string.punctuation})
    ))

    if on_ is not None:
        for k in ['note', 'velocity']:
            result[k] = int(msg[msg.rfind(k):].split(' ')[0].split('=')[1].translate(
                str.maketrans({a: None for a in string.punctuation})
            ))
    return [result, on_]


count = 0
for j, track in enumerate(mid_list[0].tracks):
    print('Track {}: {}'.format(j, track.name))
    if j == 1 or j == 2:
        for msg in track:
            print(msg2dict(msg))

'''
for m in mid_list[0].tracks[1]:
    print(m)
'''


