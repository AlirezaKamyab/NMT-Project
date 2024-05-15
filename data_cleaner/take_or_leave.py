#!/usr/bin/env python3
import os
import sys
from parsivar import Normalizer
import nltk
from deep_translator import GoogleTranslator
import time


def main():
    args = sys.argv[1:]
    try:
        translator = GoogleTranslator(source='en', target='fa')
    except:
        translator = None

    if len(args) > 0:
        filename = args[0]
    else:
        filename = input('filename >> ')

    if len(args) > 1:
        sep = args[1]
    else:
        sep = input('SEP >> ')

    if translator is not None:
        auto = input('Do you want to clean in automated mode? (Y/N)')
    else: auto = False
    if auto.lower() == 'y':
        auto = True
        new_filename = input('Filename to save to >> ')
        rest_filename = input('Save the undone (blank to not to save) >> ')
    else: auto = False

    with open(filename, 'r') as file:
        texts = file.read().split('\n')

    normalizer = Normalizer()
    tokenizer = nltk.tokenize.NLTKWordTokenizer()
    taken = []

    i = 0
    while i < len(texts):
        if texts[i].strip() == '':
            i += 1
            continue

        text = texts[i].split(sep)
        en = ' '.join(tokenizer.tokenize(text[0]))
        fa = normalizer.normalize(text[1]).replace('\u200c', ' ')

        print(f'\n\n{i + 1:<5} out of {len(texts):<5} : {len(taken):<5} Taken')
        print('SOURCE:', f'\t{en}')
        print('TARGET:', f'\t{fa}')

        if len(en) < 3 or len(fa) < 3:
            i += 1
            print('Invalid input, skipping . . .')
            continue
        if translator is not None:
            while True:
                try:
                    google = translator.translate(text[0])
                    break
                except:
                    print('Establishing connection . . .')

            if google is None:
                i += 1
                continue

            google = ' '.join(tokenizer.tokenize(google))

        if translator is not None:
            print('GOOGLE TRANSLATION:', f'\t{google}\n')
        else: print()

        if not auto:
            inp = input('ENTER to take\nG take google\nL to leave\nB to go back\nE to exit and save >> ')
        else:
            inp = 'g'
            print()

        if inp.strip() == '':
            taken.append(sep.join([en, fa]))
        if inp.strip().lower() == 'g':
            taken.append(sep.join([en, google]))

        elif inp.lower() == 'b':
            taken.pop()
            i = max(i - 1, 0)
            continue
        elif inp.lower() == 'e':
            break

        i += 1

        if i % 100 == 0:
            save(new_filename, rest_filename, taken, texts, i)
            print('Checkpoint')

    if not auto:
        new_filename = input('Filename >> ')
        rest_filename = input('Save the undone (blank to not to save) >> ')
    save(new_filename, rest_filename, taken, texts, i)

    return


def save(new_filename, rest_filename, taken, texts, i):
    if new_filename.strip() == '':
        new_filename = 'output.txt'
    with open(new_filename, 'w') as file:
        file.write('\n'.join(taken))

    if i < len(texts):
        if rest_filename.strip() == '': return
        with open(rest_filename, 'w') as file:
            file.write('\n'.join(texts[i:]))
    elif i >= len(texts):
        os.remove(rest_filename)


if __name__ == '__main__':
    main()