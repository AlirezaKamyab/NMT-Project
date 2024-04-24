# ADD LICENSE
import re
from srttime import SrtTime
import csv
import sys


def main():
    if len(sys.argv) != 3:
        en_file = input('English srt filename >> ')
        fa_file = input('Persian srt filename >> ')
    else:
        _, en_file, fa_file = sys.argv

    en_subtitles = get_subtitles(en_file)
    fa_subtitles = get_subtitles(fa_file)

    max_len = len(en_subtitles)
    en_start = 0
    fa_start = 0

    en_end = en_start + 1
    fa_end = fa_start + 1

    lst = []
    prev_op = 'n'
    while True:
        en_texts = en_subtitles[en_start:en_end]
        fa_texts = fa_subtitles[fa_start:fa_end]
        en_text = ' '.join(en['text'] for en in en_texts)
        fa_text = ' '.join(fa['text'] for fa in fa_texts)
        print(f'{en_end} out of {max_len}\nENGLISH: {en_text}\nFARSI: {fa_text}')
        inp = input("'es' skip\n'b' next line\n's' to save\n'm' merge with previous\n>> ")

        if inp == 'es':
            en_start = en_end
            fa_start = fa_end
            en_end = en_start + 1
            fa_end = fa_start + 1
            prev_op = 'es'
        elif inp == 'm' and len(lst) > 0:
            lst[-1]['en'] += f' {en_text}'
            lst[-1]['fa'] += f' {fa_text}'
            res_en = lst[-1]['en']
            res_fa = lst[-1]['fa']
            print('MERGE result:')
            print(f'ENGLISH: {res_en}\nFARSI: {res_fa}\n\n')
            en_start = en_end
            fa_start = fa_end
            en_end = en_start + 1
            fa_end = fa_start + 1
            prev_op = 'm'
        elif inp == 'b':
            en_end += 1
            fa_end += 1
            prev_op = 'b'
        elif inp == 's':
            lst.append({'en': en_text, 'fa': fa_text})
            en_start = en_end
            fa_start = fa_end
            en_end = en_start + 1
            fa_end = fa_start + 1
            prev_op = 's'

        if en_end > len(en_subtitles) or fa_end > len(fa_subtitles):
            if prev_op == 'b':
                lst.append({'en': en_text, 'fa': fa_text})
            break

    with open('output.csv', 'w') as file:
        writer = csv.writer(file)
        for sample in lst:
            writer.writerow(list(sample.values()))

    print('File saved. Checkout \'output.csv\'')


def get_subtitles(filename: str) -> list:
    with open(filename, 'r') as file:
        file = file.read()

    pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})((\n.+)*)'
    matches = re.findall(pattern, file)
    subtitles = []
    for match in matches:
        index = int(match[0])
        start = SrtTime.getStrTime(match[1])
        end = SrtTime.getStrTime(match[2])
        lines = match[3].split('\n')[1:]
        text = ' '.join(lines).strip()
        subtitles.append({'index': index, 'start': start, 'end': end, 'text': text})

    return subtitles


def get_subtitles_up_to(subtitles: list,start:SrtTime, upto:SrtTime) -> list:
    lst = []
    for sub in subtitles:
        if start <= sub['start'] < upto:
            lst.append(sub)
    return lst


if __name__ == "__main__":
    main()
