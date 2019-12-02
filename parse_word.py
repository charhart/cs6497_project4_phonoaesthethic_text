import string
import nltk
import pyphen
from functools import lru_cache
import json
import csv
from nltk.corpus import wordnet
from itertools import product
from gutenberg.acquire import load_etext
from gutenberg.cleanup import strip_headers
from statistics import mean
import ety


syllable_dict = None
cmu_phonemes = None
power_phonemes = ['L', 'M']
other_power_phonemes = ['S', 'N', 'R', 'K', 'T', 'D']
lower_power_phonemes = ['F', 'B', 'P', 'V', 'NX', 'NG', 'W', 'G', 'Z', 'SH', 'HH', 'H', 'CH', 'JH', 'Y', 'TH', 'ZH',
                        'DH']

valid_pos = ["JJ", "JJR", "JJS", "NN", "NNS", "NNP", "NNPS", "RB", "RBR", "RBS", "VB",
             "VBD", "VBG", "VBN", "VBP", "VBZ", "WRB", "WP$", "WP", "WDT", "POS", "FW",
             "MD"]
STOP = "STOP"
SIBILANT_AFFRICATE = "SIBILANT_AFFRICATE"
SIBILANT_FRICATIVE = "SIBILANT_FRICATIVE"
FRICATIVE = "FRICATIVE"
TAP_FLAP = "TAP_FLAP"
LATERAL_APPROXIMANT = "LATERAL_APPROXIMANT"
NASAL = "NASAL"
NON_SIBILANT_FRICATIVE = "NON_SIBILANT_FRICATIVE"
APPROXIMANT = "APPROXIMANT"
TRILL = "TRILL"
LATERAL_FRICATIVE = "LATERAL_FRICATIVE"
LATERAL_TAP_FLAP = "LATERAL_TAP_FLAP"
manners = {
    "B": STOP,
    "CH": SIBILANT_AFFRICATE,
    "D": STOP,
    "DH": FRICATIVE,
    "DX": TAP_FLAP,
    "EL": LATERAL_APPROXIMANT,
    "EM": NASAL,
    "EN": NASAL,
    "F": NON_SIBILANT_FRICATIVE,
    "G": STOP,
    "JH": SIBILANT_AFFRICATE,
    "K": STOP,
    "L": LATERAL_APPROXIMANT,
    "M": NASAL,
    "N": NASAL,
    "NX": NASAL,
    "NG": NASAL,
    "P": STOP,
    "Q": STOP,
    "R": APPROXIMANT,
    "S": SIBILANT_FRICATIVE,
    "SH": SIBILANT_FRICATIVE,
    "T": STOP,
    "TH": NON_SIBILANT_FRICATIVE,
    "V": NON_SIBILANT_FRICATIVE,
    "W": APPROXIMANT,
    "WH": APPROXIMANT,
    "Y": APPROXIMANT,
    "Z": SIBILANT_FRICATIVE,
    "ZH": SIBILANT_FRICATIVE
}
short_vowels = ['AE', 'AH', 'AO', 'AX', 'EH', "IH", "UH"]
long_vowels = ['AA', 'AY', 'EY', "IY", "OW", "UW"]
other_vowels = ['AW', "AXR", "ER", "IX", "OY", "UX"]

high_vowels = ["IH", "IX", "IY", "UH", "UW", "UX"]
mid_vowels = ["AO", "AX", "EH", "EY", "OY", "OW"]
low_vowels = ["AA", "AE", "AH", "AY", "AW"]

front_vowels = ["AE", "EH", "IH", "IY", "EY"]
central_vowels = ["AA", "AH", "AX", "IX", "UX", "AY", "AW", "OW"]
back_vowels = ["AO", "UH", "UW", "OY"]

most_common_words = list()


def write_header(writer):
    writer.writerow(['#', 'word', 'score', 'origin',
                     'syllables', 'best_synonym', 'best_synonym_score', 'synonym_wins', 'synonym_improvement',
                     'pos', 'articulations',
                     'three_plus_syllables', 'stress_on_first',
                     'uses_m', 'uses_l', 'uses_good_consonants', 'no_weak_consonants',
                     'three_manners_of_articulation', 'short_vowels_only', 'front_vowels_first',
                     'low_vowels_first'])


def write_row(writer, rank, word, stats):
    out = list()
    out.append(str(rank))
    out.append(word)
    if stats:
        score = stats.get('score_percent', -1.0)
        out.append(str(stats.get('score_percent', -1.0)))
        out.append(str(stats.get('origin')))
        out.append(stats.get('syllable_count', 0))

        pos = stats.get('pos', 'UK')
        best_score = -1.0
        best_score_i = 0
        syns = stats.get('synonyms', list())
        syn_scores = stats.get('synonym_scores', list())
        if len(syns) > 0:
            for i in range(len(syns)):
                syn_score = int(syn_scores[i])
                syn = syns[i]
                syn_pos = get_pos_tag(syn)

                if pos == syn_pos and syn_score > best_score:
                    best_score_i = i
                    best_score = syn_score

            if best_score > -1.0:
                out.append(syns[best_score_i])
                out.append(syn_scores[best_score_i])
                out.append(str(best_score > score))
                out.append(str(best_score - score))
            else:
                out.append('')
                out.append('-1.0')
                out.append('False')
                out.append('0.0')
        else:
            out.append('')
            out.append('-1.0')
            out.append('False')
            out.append('0.0')
        out.append(pos)
        articulations = stats.get('manners_of_articulation', list())
        out.append("|".join(articulations))
        out.append(stats.get('more_than_three_syllables', False))
        out.append(stats.get('stress_on_first', False))
        out.append(stats.get('contains_m', False))
        out.append(stats.get('contains_l', False))
        out.append(stats.get('contains_other_strong_consonants', False))
        out.append(stats.get('no_weak_consonants', False))
        out.append(stats.get('gt3_manners_of_articulation', False))
        out.append(stats.get('only_short_vowels', False))
        out.append(stats.get('more_front_vowels_than_not', False))
        out.append(stats.get('more_low_vowels_than_not', False))
    writer.writerow(out)


def common_words():
    print("loading common words....")
    with open('./data/common_english_words.txt', 'r') as f:
        ln = f.readline()
        n = 0
        with open('./data/common_scores.csv', mode='w') as out_file:
            writer = csv.writer(out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            write_header(writer)
            while ln:
                global most_common_words
                ln = ln.replace('\n', '').lower().strip()
                if ln.startswith('#'):
                    continue
                most_common_words.append(ln)
                w_stats = word_stats(ln)
                if w_stats:
                    write_row(writer, n, ln, w_stats)
                ln = f.readline()
                n += 1


def setup(get_common_words=True):
    print("doing setup...")
    pyphen.language_fallback('en_GB')

    # http://www.gutenberg.org/ebooks/3204
    global syllable_dict
    syllable_dict = dict()
    with open('./data/mhyph.txt', 'r') as f:
        ln = f.readline()
        while ln:
            ln = ln.replace('\n', '')
            final_s = list()
            word = ln.replace('-', '').lower()
            spl = ln.split('-')
            for s in spl:
                spaces = s.split(' ')
                final_s.extend(spaces)
            syllable_dict[word] = final_s
            ln = f.readline()

    global cmu_phonemes
    try:
        cmu_phonemes = nltk.corpus.cmudict.dict()
    except LookupError:
        nltk.download('cmudict')
        cmu_phonemes = nltk.corpus.cmudict.dict()
    if get_common_words:
        common_words()


@lru_cache(maxsize=1000000)
def get_pos_tag(word):
    tokens = nltk.word_tokenize(word)
    tokens_out = nltk.pos_tag(tokens)
    if len(tokens_out) == 0:
        return "UNKNOWN"
    else:
        return tokens_out[0][1]


@lru_cache(maxsize=1000000)
def get_synonyms(word):
    word = word.strip().lower()
    synonyms = []
    antonyms = []

    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            syn = l.name().lower().strip()
            if syn != word:
                if '_' not in syn:
                    synonyms.append(syn)
            if l.antonyms():
                ant = l.antonyms()[0].name().lower()
                if '_' not in ant:
                    if word != ant:
                        antonyms.append(ant)


    return list(set(synonyms)), list(set(antonyms))


@lru_cache(maxsize=1000000)
def phonemes(word, syllables=None):
    if not syllables:
        syllables = list()

    output = list()
    stress = list()
    stress_weight = list()
    vowels = list()
    p_list = cmu_phonemes.get(word.lower(), [[]])
    if len(p_list) > 0:
        p_list = p_list[0]
    for p in p_list:
        output.append(p)
        if p[-1].isdigit():
            stress.append(p)
            stress_weight.append(int(p[-1]))
            vowels.append(p[:-1])
        else:
            stress.append('')
            stress_weight.append(-1)

    return output, stress, stress_weight, vowels


@lru_cache(maxsize=1000000)
def syllables_in_word(word, lang='en_US'):
    total_syllables = list()
    for w in word.split(' '):
        if w.lower() in syllable_dict:
            total_syllables.extend(syllable_dict.get(w.lower()))
        else:
            d = pyphen.Pyphen(lang=lang)
            hyp = d.inserted(w)
            s = list(hyp.split('-'))
            total_syllables.extend(s)
            syllable_dict[w.lower()] = s

    return total_syllables


@lru_cache(maxsize=100000)
def word_origin(word):
    origin = ety.origins(word)
    if len(origin) == 0:
        return "Undefined"
    lang = origin[0].language.name.split('(')[0].strip()
    return lang


@lru_cache(maxsize=1000000)
def word_stats(word, get_synonym_scores=True):
    word = word.lower().strip()
    stats = dict()
    stats['syllables'] = syllables_in_word(word)
    ph, stresses, stress_weights, vowels = phonemes(word)
    pos = get_pos_tag(word)
    if pos not in valid_pos:
        return None
    if len(ph) == 0:
        return None
    stats['origin'] = word_origin(word)
    stats['phonemes'] = ph
    stats['stress'] = stresses
    stats['stress_weights'] = stress_weights
    synonyms, _ = get_synonyms(word)
    stats['synonyms'] = synonyms
    synonym_scores = list()
    if get_synonym_scores:
        for s in synonyms:
            s_stats = word_stats(s, get_synonym_scores=False)
            if s_stats:
                synonym_scores.append(s_stats.get('score_percent'))
            else:
                synonym_scores.append(-1.0)
    stats['synonym_scores'] = synonym_scores
    stats['syllable_count'] = len(stats['syllables'])
    stats['vowels'] = vowels
    stats['pos'] = pos

    score = 0
    if stats['syllable_count'] > 2:
        stats['more_than_three_syllables'] = True
        score += 1

    found_weights = 0
    if stats['syllable_count'] > 1:
        for s in stats['stress_weights']:
            if s > -1:
                found_weights += 1
                if found_weights == 1 and s == 1:
                    stats['stress_on_first'] = True
                    score += 1

    for p in product(power_phonemes, stats['phonemes']):
        if p[0] == p[1]:
            key = "contains_{}".format(p[0].lower())
            if key not in stats:
                stats[key] = True
                score += 1

    for p in product(other_power_phonemes, stats['phonemes']):
        if p[0] == p[1]:
            key = "contains_other_strong_consonants"
            if key not in stats:
                stats[key] = True
                score += 1

    for p in product(lower_power_phonemes, stats['phonemes']):
        if p[0] == p[1]:
            key = "contains_weak_consonants"
            if key not in stats:
                stats[key] = True

    if not stats.get('contains_weak_consonants', None):
        stats['no_weak_consonants'] = True
        score += 1

    manners_of_articulation = set()
    for p in stats['phonemes']:
        m = manners.get(p)
        if m:
            manners_of_articulation.add(m)
    stats['manners_of_articulation'] = list(manners_of_articulation)
    stats['gt3_manners_of_articulation'] = len(manners_of_articulation) > 2
    if stats['gt3_manners_of_articulation']:
        score += 1

    only_short_vowels = True
    front_vowel_count = 0
    non_front_vowel_count = 0
    low_vowel_count = 0
    non_low_vowel_count = 0
    for v in vowels:
        if v in long_vowels or v in other_vowels:
            only_short_vowels = False
        if v in front_vowels:
            front_vowel_count += 1
        else:
            non_front_vowel_count += 1
        if v in low_vowels:
            low_vowel_count += 1
        else:
            non_low_vowel_count += 1

    if front_vowel_count > non_front_vowel_count:
        stats['more_front_vowels_than_not'] = True
        score += 1
    else:
        stats['more_front_vowels_than_not'] = False

    if low_vowel_count > non_low_vowel_count:
        stats['more_low_vowels_than_not'] = True
        score += 1
    else:
        stats['more_low_vowels_than_not'] = False

    if only_short_vowels:
        score += 1

    stats['front_vowel_count'] = front_vowel_count
    stats['non_front_vowel_count'] = non_front_vowel_count
    stats['low_vowel_count'] = low_vowel_count
    stats['non_low_vowel_count'] = non_low_vowel_count

    stats['only_short_vowels'] = only_short_vowels
    stats['score'] = score

    score_max = 10.0
    stats['score_percent'] = ((score * 1.0)/score_max) * 100

    return stats


def get_gutenberg_text(id):
    try:
        text = strip_headers(load_etext(id)).strip()
        return text
    except Exception as ex:
        print(ex)
    return ''


def get_average_score(txt):
    if len(txt) == 0:
        return None
    try:
        stripped = txt.translate(str.maketrans('', '', string.punctuation))

        spl = stripped.split()
        scores = list()
        for s in spl:
            if len(s) > 0:
                stats = word_stats(s)
                if stats:
                    score = stats.get('score_percent', -1.0)
                    if score >= 0:
                        scores.append(score)

        return len(spl), mean(scores)
    except Exception as ex:
        print(ex)
    return None


def get_from_file(f):
    with open('./data/{}.txt'.format(f), 'r') as file:
        txt = file.read().replace('\n', '')
    return txt


if __name__ == """__main__""":
    if not syllable_dict:
        setup(get_common_words=False)

    # test_words = ['aesthetics', 'luminous', 'tremulous', 'alumnus', 'wapping', 'flatulent', 'gripe', 'zoo', 'tart']
    test_words = []
    for t in test_words:
        sts = word_stats(t)
        if sts:
            print(json.dumps(sts, indent=4, sort_keys=True))

    print('scoring works')
    with open('./data/text_scores.csv', mode='a') as out_file:
        writer = csv.writer(out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # writer.writerow(["name", "author", "length", "score"])
        works = [
            # ('sonnets', 'shakespeare', 1041)
            # ('leaves_of_grass', 'whiteman', 1322),
            # ('odyssey', 'homer', 1727),
            # ('canterbury_tales', 'chaucer', 22120),
            # ('faust', 'goethe', 14591)
        ]
        files = [
            # ('poemsofemilydickinson', 'poems_of_emily_dickinson', 'dickison'),
            # ('SILMARILLION', 'silmarillion', 'tolkien'),
            # ('the_hobbit', 'the_hobbit', 'tolkien'),
            # ('gettysburg', 'gettysburg_address', 'lincoln'),
            # ('lotr', 'lotr', 'tolkien')

        ]

        for w in works:
            print('evaluating {}'.format(w[0]))
            t = get_average_score(get_gutenberg_text(w[2]))
            writer.writerow([w[0], w[1], t[0], t[1]])
            out_file.flush()

        for f in files:
            print('evaluating {}'.format(f[1]))
            t = get_average_score(get_from_file(f[0]))
            writer.writerow([f[1], f[2], t[0], t[1]])
            out_file.flush()

        with open('./data/gutenberg_top_100.txt') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                book = row[0].replace(" ", "_").lower().strip().replace(';', '').replace(',', '').replace('.', '')\
                    .replace(':', '')
                name = row[1].replace(" ", "_").lower().strip().split('_')[-1].replace('"', '')
                code = (row[2].replace('"', '').strip())
                code = code.translate(str.maketrans('', '', string.punctuation))
                print('evaluating {}'.format(book))
                t = get_average_score(get_gutenberg_text(int(code)))
                if t:
                    writer.writerow([book, name, t[0], t[1]])
                    out_file.flush()
