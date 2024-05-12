#!/usr/bin/env python3
import sys
from parsivar import Normalizer
import nltk


def main():
    args = sys.argv[1:]

    if len(args) > 0:
        filename = args[0]
    else:
        filename = input('filename >> ')

    if len(args) > 1:
        sep = args[1]
    else:
        sep = input('SEP >> ')

    with open(filename, 'r') as file:
        texts = file.read().split('\n')

    normalizer = Normalizer()
    tokenizer = nltk.tokenize.NLTKWordTokenizer()
    taken = []

    i = 0
    while i < len(texts):
        text = texts[i].split(sep)
        en = ' '.join(tokenizer.tokenize(text[0]))
        fa = normalizer.normalize(text[1]).replace('\u200c', ' ')
        print(f'\n\n{i + 1:<5} out of {len(texts):<5} : {len(taken):<5} Taken')
        print('SOURCE:', f'\t{en}')
        print('TARGET:', f'\t{fa}')
        inp = input('ENTER to take\nL to leave\nB to go back\nE to exit and save >> ')
        if inp.strip() == '':
            taken.append(sep.join([en, fa]))
        elif inp.lower() == 'b':
            taken.pop()
            i = max(i - 1, 0)
            continue
        elif inp.lower() == 'e':
            break

        i += 1

    new_filename = input('Filename >> ')
    if new_filename.strip() == '':
        new_filename = 'output.txt'
    with open(new_filename, 'w') as file:
        file.write('\n'.join(taken))

    if i < len(texts):
        rest_filename = input('Save the undone (blank to not to save) >> ')
        if rest_filename.strip() == '': return
        with open(rest_filename, 'w') as file:
            file.write('\n'.join(texts[i:]))
    return


if __name__ == '__main__':
    main()