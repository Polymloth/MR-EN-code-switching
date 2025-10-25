import re
from wordfreq import zipf_frequency
import langid
import time
import statistics
from collections import defaultdict

# this is the most inefficient and mathematically weak code-switching detector ever. You're welcome :)

#######################################################################################################################
################################################# manual fix sets #####################################################
#######################################################################################################################

NOT_ENGLISH = {'tar', 'jar', 'kai', 'la', 'na', 'mi', 'ali', 'te', 'ne', 'ya',
                'th', 'anna', 're', 'de', 'ha', 'til', 'co', 'jo', 'wa', 'ka',
                'le', 'ab', 'ta', 'rt', 'va', 'ho', 'ma', 'sri', 'li', 'aa', 'si', 
                'tai', 'ni', 'ch', 'tr', 'ye', 'ss', 'bo', 'ti', 'pan', 'dharma',
                'ch', 'key', 'chi', 'tar', 'jo', 'ch', 'mat', 'kay', 'mag', 'fir',
                'war', 'cr', 'bagh', 'shah', 'java', 'jay', 'ku', 'jee', 'bal', 
                'nigh', 'che', 'don', 'nai', 'atta', 'da', 'khan', 'bolo', 'ja',
                'par', 'abe', 'mg', 'hoy', 'jana', 'bari', 'bey', 'rey', 'ga', 'sop',
                'rok', 'nay', 'patti', 'hee', 'dana', 'taras', 'mast', 'zing', 'mor',
                'dei', 'hum', 'nae', 'sunder', 'marr', 'bahi', 'lah', 'bas', 'sod', 
                'hafta', 'mal', 'maine', 'aya', 'pta', 'meg', 'gai', 'abbe', 'bolet',
                'santo', 'alla', 'pad', 'hume', 'mana', 'salle', 'aye', 'di', 'shakya',
                'bani', 'yr', 'mama', 'mem', 'ja', 'barr', 'ann', 'asch', 'nigh', 
                'pi', 'jagan', 'ada', 'nai', 'da', 'han', 'krupa', 'dey', 'koan',
                'atta', 'hal', 'budd', 'tass', 'bich', 'sta', 'katya', 'maze', 'rd',
                'tau', 'kitti', 'kel', 'pah', 'dag', 'sh', 'gal', 'marin', 'neil',
                'shu', 'maht', 'zara', 'kok', 'shan', 'swami', 'jai', 'didi', 'sari',
                'sultan', 'basel', 'jap', 'der', 'ana', 'veer', 'sor', 'hath', 'marathi',
                'hindu','dole', 'pane', 'hone', 'toch', 'wo', 'tel', 've', 'pa', 'nam', 
                'thet', 'mere', 'kali', 'bazaar', 'ba', 'em', 'pal' 'sho', 'nan', 'san',
                'ad', 'pith', 'gala', 'bore', 'bg', 'en', 'pak', 'hind', 'ml', 'koop'
                'las', 'tat', 'al', 'ra', 'sur', 'mali', 'po', 'el', 'dost', 'mare', 
                'galli', 'pas', 'whee', 'das', 'ol', 'sut', 'gore', 'ca', 'mann', 'sap',
                'pap', 'lat', 'hue', 'mans', 'mains', 'tara', 'lena', 'sha', 'hon', 
                'ajar', 'une', 'maharashtra', 'maharashtrat', 'maharashtracha', 'se',
                'maharashtrachya', 'maharashtrian', 'maharastra', 'hare', 'ram', 'lo',
                'bete', 
                }

# 'en' 'el' and 'al' are SPANISH !

ENGLISH = {'wud', 'assemb', 'partiality', 'luckyyy', 'ur', 'gif', 'vip', 'sir', 'celebrity',
           'dance', 'free', 'supreme', 'respect', 'power', 'chinese', 'acting', 'music',
           'paper', 'also', 'hotel', 'tweeter', 'joyful', 'exams', 'farmar', 'congress',
           'lockdown', 'recipes', 'corona', 'centre', 'police', 'dicel', 'exactly', 'flop',
           'receipe', 'bc', 'nice', 'congree', 'covid', 'congratulations', 'clic', 'coz',
           'election', 'recipi', 'coment', 'corts', 'public', 'correct', 'companya', 
           'court', 'recepi', 'acount', 'helicopter', 'cake', 'actually', 'sargical',
           'contestant', 'central', 'recepie', 'excellent', 'diarect', 'comment', 'wite',
           'vidio', 'Govt', 'castes', 'veg', 
           }

# can't appear alone but acceptable in an English phrase
UNIGRAM_BAN = {'he', 'do', 'to', 'ant', 'hi', 'ya', 'kay', 'mat', 'pan', 'war', 'gram',
               'me', 'mars', 'pun', 'sang', 'nay', 'mule', 'to', 'jan', 'gap', 'kate',
               'lag', 'mile', 'gel', 'am', 'rag', 'moth', 'for', 'are', 'as', 'did',
               'tan', 'an', 'chop', 'mug', 'ham', 'tin', 'lay', 'deed', 'go', 'utter',
               'main', 'yet', 'saga', 'mud', 'dish', 'pure', 'hey', 'mash', 'my', 'on',
               'man', 'band', 'ache', 'pot', 'mate', 'mule', 
               } 

NE_LIST = {'delhi', 'mumbai', 'monica', 'sweden', 'nathan', 'allah', 'stalin', 'andrew',
           'jane', 'sarah', 'ibrahim', 'hamilton', 'linda', 'hitler', 'john', 'buddha',
           'alia', 'pam', 'russia', 'sara', 'bangkok', 'wayne', 'tim', 'europe', 'germany',
           'newcastle', 'russell', 'bali', 'oscar', 'india', 'liverpool', 'britain', 'paris',
           'gregory', 'marcus', 'michael', 'goan', 'goa', 'bombay', 'beth', 'california', 
           'bobby', 'manchester', 'islam', 'peterson', 'carolina', 'sahara', 'smith', 'italia',
           'france', 'cyrus', 'angeles', 'ronald', 'mira', 'korea', 'phil', 'scotland',
           'thailand', 'portugal', 'murray', 'jawaharlal', 'israel', 'thackeray', 'peru',
           'trump', 'syria', 'america', 'garcia', 'muslim', 'romeo', 'antonio', 'chaplin',
           'marx', 'vincent', 'arab', 'anderson', 'londo', 'norway', 'hernandez', 'thor',
           'clinton', 'robert', 'iran', 'baltimore', 'persia', 'oskar', 'rafael', 'jessica',
           'nagpur', 'gujarat', 'pakistan', 'kolhapur', 'jalgaon', 'ajit', 'rao', 'bihar',
           'raja', 'ganesh', 'shri', 'youtube', 'twitter', 'tweet', 'google', 'cm', 'pune',
           'chatrapati', 'rohit', 'modiji', 'modi', 'rajesh', 'bollywood', 'deshmukh', 'hm',
           'solapur',  'ed', 'psi', 'sam', 'ben', 'dubai', 'sri', 'amazon', 'Soniya', 'paul',
           'scindia', 'ncp', 'bbc', 'krishna', 'gop', 'anita', 'punjab'} 


#######################################################################################################################
################################################ TEXT PROCESSING ######################################################
#######################################################################################################################

# precompiling regex patterns
dots_pattern = re.compile(r'\.{2,}')
abbr_pattern = re.compile(r'(?:(?<=\W)|^)[A-Za-z]\.(?: )?(?:[A-Za-z](?:\. ?| ))+')
clean_abbr_pattern = re.compile(r'[ .]')
split_pattern = re.compile(r'([.,!?…";:*/])|\s+') # preserve "don't"
part_pattern = re.compile(r'^[^a-zA-Z]+|[a-zA-Z]+[^a-zA-Z]*[a-zA-Z]+|[^a-zA-Z]+$')
nonletter_pattern  = re.compile(r'[^a-zA-Z]+')
dev_pattern = re.compile(r'[\u0900-\u097F]')
strip_pattern = re.compile(r'^[^a-zA-Z]+|[^a-zA-Z]+$')      

def process_sentence(sentence: str):
    sentence = sentence.replace('@USER', '') 
    sentence = sentence.replace('&amp;', '') 
    sentence = sentence.replace('&quot;', '')
    sentence = dots_pattern.sub('.', sentence) # normalising strings of dots ....
    sentence = abbr_pattern.sub(lambda m: f" {clean_abbr_pattern.sub('', m.group()).upper()} ", sentence) # normalising abbreviations
    # DEFINE MORE FIXES HERE IF NEEDED
    
    return sentence.strip(' \"\'')

def split_sentence(sentence):
    tokens = []
    parts = split_pattern.split(sentence) 
    parts = [p for p in parts if p] # remove empties
    for part in parts:
        subparts = part_pattern.findall(part)
        if not subparts:  # if the token has no letters at all
            subparts = nonletter_pattern.findall(part)
        tokens.extend(subparts)
    return tokens

def process_token(token: str):
    token = 'dev_token' if dev_pattern.search(token) else token    
    token = strip_pattern.sub('', token)
    # DEFINE MORE FIXES HERE IF NEEDED

    return token.strip('\'')

def removal_check(processed_sentence:str):
    latin_count = sum('A' <= ch <= 'Z' or 'a' <= ch <= 'z' for ch in processed_sentence)
    if latin_count < 10:
        return True
    return False

#######################################################################################################################
#################################################### LANG SCORING #####################################################
#######################################################################################################################

def foreign_check(sentence: str):
    raw_tokens = split_sentence(sentence)
    sent = ' '.join([process_token(token) for token in raw_tokens if process_token(token)])

    first = ['__label__fr', '__label__es']
    second = ['__label__it', '__label__fr', '__label__es']
    labels, probs = model.predict(sent, k=2)
    probs = tuple(float(p) for p in probs) # numpy scalars did weird things in the if statement
    # either the first one is french or spanish and the prob is > 0.8 or the second is french, spanish or italian and their combined probability is > 0.8, with more weight on the first one
    if labels[0] in first and len(sent) > 27: # length limit for prediction accuracy
        if len(labels) == 2 and len(probs) == 2:
            if probs[0] > 0.8 or (labels[1] in second and probs[0] > 0.55 and probs[1] > 0.25):
                return True
    return False

'''
nltk.download('wordnet') # if necessary
from nltk.corpus import wordnet
def wordnet_check(word):
    return bool(wordnet.synsets(word))
'''

#########################################################################
# FASTTEXT

import fasttext
model = fasttext.load_model(r'/home/loth/Projects/fastext_model_lid_176')

def ftt_check(text):
    labels, probs = model.predict(text.lower(), k=176)
    results = dict(zip(labels, probs))
    langs = ['en', 'de', 'es', 'fr', 'mr', 'hi']
    probs = {lang: results.get(f'__label__{lang}', 0.0) for lang in langs}
    combined = probs['en']+probs['fr']+probs['es']+probs['de']
    return combined

#########################################################################
# DICTIONARIES

def load_wordlist(path):
    words = set()
    with open(path, encoding='utf-8') as f:
        for line in f:
            word = line.strip()
            if word:
                words.add(word)
    return words

SCOWL_WORDS = load_wordlist('Source-files/scowl.txt')

def scowl_check(tokens):
    falses = 1
    fraction = 1/len(tokens)
    for token in tokens:
        if token.lower() not in SCOWL_WORDS or manual_override(token):
            falses -= fraction
    return falses

SEMCOR_WORDS = load_wordlist('Source-files/semcor.txt')

def semcor_check(ngram: str):
    falses = 1
    fraction = 1/len(ngram)
    for token in ngram:
        if token.lower() not in SEMCOR_WORDS:
            falses -= fraction
    return falses

#########################################################################
# LANGID

def langid_check(text):
    data = langid.rank(text)
    labels = [l for (l, _) in data]
    probs = [p for (_, p) in data]
    en_index = labels.index('en')
    fr_index = labels.index('fr')
    avg_en = int(min(en_index, (en_index+fr_index)/2))
    flag = probs[avg_en] > 0
    return flag

#########################################################################
# ZIPF

def average_zipf(ngram):
    length = len(ngram)
    zipf_sum = 0
    for token in ngram:
        zipf_sum += zipf_frequency(token, 'en')
    return zipf_sum / length

#########################################################################

def manual_override(token):
    word = token.lower()
    if word in NOT_ENGLISH or (len(word) == 1 and word not in {'a', 'i', 'u', 'y', 'w', 'r'}): # overriding some word tags that might falsely be en
        return 0
    elif word in ENGLISH or word in SEMCOR_WORDS:
        return 1
    return None

def total_score(token, ngram_scores: dict):
    ftt = ngram_scores['ftt']
    #zipf = ngram_scores['zipf']
    #poslangid = ngram_scores['lagid>0']
    scowl = ngram_scores['scowl%']
    semcor_r = ngram_scores['semcor%']
    
    scowl_scaled = (ftt+scowl**2)/2
    semcor_scaled = (scowl_scaled+semcor_r)/2

    override = manual_override(token)
    total = override if override is not None else semcor_scaled

    return min(1, total)

def token_scores(token:list):
    score = {'ftt': ftt_check(token), 'scowl%': scowl_check([token]), 'semcor%': semcor_check([token])} # 'zipf': average_zipf(tokens), 'lagid>0': langid_check(ngram),
    return score

#######################################################################################################################
#################################################### NER FUNCTIONS ####################################################
#######################################################################################################################

def simple_NER(tokens):
    ne_mask = []
    for i in range (0,len(tokens)):
        boolie = False

        # if a short string of uppercase letters, possibly an abbreviation, which is not surrounded by other uppercase strings
        if tokens[i].isupper() and len(tokens[i]) < 4 and tokens[i] != 'LOL':
            prev = tokens[i-1] if i > 0 else ''
            next_ = tokens[i+1] if i < len(tokens)-1 else ''
            if not (prev.isupper() or next_.isupper()):
                boolie = True

        if tokens[i].lower() in NE_LIST:
            boolie = True
        
        ne_mask.append(boolie)
    return ne_mask

def normalise_ne(token):
    token = token.upper() if len(token) < 4 else token
    if not token.isupper() or len(token) > 3 or token.lower() in ['rao', 'goa', 'sri', 'tim']: # don't change all uppercase
        token = token[0].upper() + token[1:].lower()
    return token


#######################################################################################################################
################################################### MAIN FUNCTIONS ####################################################
#######################################################################################################################

'''
def get_arrays(tokens:list): # for more accurate results with ngrams, started with this but dropped the idea
    def ngrams(k):
        return {i: (tokens[i:i+k], token_scores(tokens[i:i+k])) for i in range(len(tokens)-k+1)}
    arrays = { 
        1: ngrams(1),
        2: ngrams(2),
        3: ngrams(3),
        4: ngrams(4)
    }
    return arrays
'''

map_to_tabs = {
    'NE': '\t-\t-\tNE\t\n',
    'MR': '\t-\tMR\t-\t\n',
    'EN': '\tEN\t-\t-\t\n',
}

def generate_n_random_file(corpus:list, n:int):
    random_set = set(corpus)
    output_list = [] 
    i = 0
    break_flag = False
    s_count = 0
    for sentence in random_set:
        s_count += 1
        if break_flag == True:
            break
        output, _, _ = analyse_sentence(sentence)
        tokens = output['word_tokens']
        mask = output['mask']
        for tuple_ in zip(tokens, mask):
            if tuple_[0] == 'dev_token':
                continue
            
            if i >= n:
                break_flag = True
                break             
            if tuple_[1] != 'NE':    
                i += 1
                output_list.append(tuple_)

    with open(f'{n}-randoms-no-NEs.tsv', 'w', encoding='utf-8') as randoms:
        randoms.write(f'{s_count} sentences up to {n} tokens\n')
        for token, tag in output_list:
            suffix = '\t\t\t\t\n' if token == 'dev_token' else map_to_tabs[tag]
            randoms.write(token + suffix)
    return

def norm_n_sort(d:dict):
    normalised = {k: v for k, v in d.items() if v}
    return dict(sorted(normalised.items(), key=lambda x: (-x[1], x[0])))

def find_engrams(zipped_tokens, totals):
    ngrams = []
    ngram_indices = []
    # temps:
    current_ngram = []
    current_indices = []

    for index, token, ne in zipped_tokens:
        if token == 'dev_token':
            continue
        score = 0 if ne else totals[index] # exclude NEs
        if score > 0.7:
            # still inside an English segment
            current_ngram.append(token)
            current_indices.append(index)
        else:
            if current_ngram:
                # only append if it's not a banned unigram or
                if not (len(current_ngram) == 1 and current_ngram[0].lower() in UNIGRAM_BAN):
                    ngrams.append(current_ngram)
                    ngram_indices.append(current_indices)
                current_ngram, current_indices = [], []


    if current_ngram:
        if not (len(current_ngram) == 1 and current_ngram[0].lower() in UNIGRAM_BAN):
            ngrams.append(current_ngram)
            ngram_indices.append(current_indices)
        current_ngram, current_indices = [], []


    # Filter en_words as well so the banned unigrams aren't counted individually
    #en_words = [w for w in en_words if w not in UNIGRAM_BAN]
    en_words = []
    en_indeces = set()
    for (ngram, indeces) in zip(ngrams, ngram_indices):
        for word, index, in zip(ngram, indeces):
            en_words.append(word)
            en_indeces.add(index)

    return en_words, en_indeces, ngrams, ngram_indices

# some global sets to be updated within the main corpus processing function -- not really "good practice" but whatevs
en_type_dict = defaultdict(int)
mr_type_dict = defaultdict(int)
ne_type_dict = defaultdict(int)
ngram_dict = {i: defaultdict(int) for i in range(1,250)}
ngramlens = defaultdict(int)
en_ratios = [] # en / words
ratio_dict = defaultdict(int)
lenlist = []
#only_english = []
#only_mixed = []

def analyse_sentence(raw_sentence:str):
    sentence = process_sentence(raw_sentence) # 0.5e-05
    raw_tokens = split_sentence(sentence) # 2.07e-05
    processed_tokens = [process_token(token) for token in raw_tokens if process_token(token)] # 2.96e-05 + 0.35e-05
    
    zipped_tokens = []
    NEs = []
    word_tokens = []
    for i, (token, ne) in enumerate(zip(processed_tokens, simple_NER(processed_tokens))):
        zipped_tokens.append((i, token, ne))
        if ne:
            NEs.append((i,normalise_ne(token)))
            
        else:
            word_tokens.append((i,token))

    totals = {}
    for (i, token) in word_tokens: # 0.01216s = 1216e-05 -> 0.0005655s = 56e-05
        if token == 'dev_token':
            continue
        else:
            scores = token_scores(token)
            total = total_score(token, scores)
            totals[i] = total

    en_words, en_indeces, ngrams, ngram_indeces = find_engrams(zipped_tokens, totals) # 0.25e-05

    words_len = len(word_tokens)+0.00000001
    en_len = len(en_words)
    ne_len = len(NEs)

    mask = []
    for (i, token, ne) in zipped_tokens:
        if token == 'dev_token':
            mask.append(token)
        elif ne:
            mask.append('NE')
        elif i in en_indeces:
            mask.append('EN')
        else:
            mask.append('MR')

    data_fields = {
                    'sentence': raw_sentence,
                    'all_tokens': raw_tokens,
                    'mask': mask,
                    'word_tokens': processed_tokens,
                    'words_len': len(processed_tokens),
                    'NEs': [token for i, token in NEs],
                    'NE_len': ne_len,
                    'non-NE_tokens': [token for i, token in word_tokens],
                    'non-NE_len': words_len,
                    'en_words': en_words,
                    'en_len': en_len,
                    'en_ratio': round(en_len/words_len,3), 
                    'scores': totals,
                    'ngrams': ngrams,
                    'ngram_indeces': ngram_indeces,
                    }
    
    sentence_data = {}
    for field, value in data_fields.items():
        sentence_data[field] = value

    if words_len != 0:
        en_ratios.append(en_len/words_len)
        ratio = round(en_len/words_len,3)
        ratio_dict[ratio] += 1
        #if ratio != float(0):
        #    only_mixed.append(raw_sentence) # filtering out english sentences

    for ngram in ngrams:
        ngramlens[len(ngram)] += 1
        lenlist.append(len(ngram))
        if ngram:
            string = ' '.join(ngram)
            ngram_dict[len(ngram)][string.lower()] += 1

    for i, token, ne in zipped_tokens:
        if token in en_words:
            en_type_dict[token.lower()] += 1
        elif ne:
            ne_type_dict[normalise_ne(token)] += 1
        else:
            mr_type_dict[token.lower()] += 1

    return sentence_data, int(words_len+ne_len), en_len

def process_corpus(corpus: list):
    all_sentence_data = {}
    word_tokens = 0
    en_tokens = 0

    for i, raw_sentence in enumerate(corpus):
        if i % 1000 == 0:
            print(f'\r{i}', end='')

        sentence_data, words_len, en_len = analyse_sentence(raw_sentence)
        word_tokens += words_len
        en_tokens += en_len
        all_sentence_data[i] = sentence_data

    return all_sentence_data, word_tokens, en_tokens

#######################################################################################################################
######################################################## MAIN #########################################################
#######################################################################################################################

start = time.time()
with open('Source-files/only_mixed_language.csv', 'r', encoding='utf-8') as f: 
    corpus = [line for line in f.readlines()]

#generate_n_random_file(corpus,1000)
#quit()

data, word_tokens, en_tokens = process_corpus(corpus) # 45/27 minutes for a full run (filtered/only-mixed)

normalised_ngrams = {k: dict(v) for k, v in ngram_dict.items() if dict(v)}
sorted_ngrams = {k: dict(sorted(v.items(), key=lambda x: -x[1])) for k, v in sorted(normalised_ngrams.items(), key=lambda x: x[0])}
ngramlens = dict(sorted(ngramlens.items(), key=lambda x: x[0]))
ratio_dict = dict(sorted(ratio_dict.items(), key=lambda x: x[0]))

PREFIX = 'Mixed-lang-only/MULTILINGUAL-ONLY'

file_dict = {f'{PREFIX}-en-types': norm_n_sort(en_type_dict), 
             f'{PREFIX}-mr-types': norm_n_sort(mr_type_dict), 
             f'{PREFIX}-NE-types': norm_n_sort(ne_type_dict),
             f'{PREFIX}-1-grams': norm_n_sort(ngram_dict[1]),
             f'{PREFIX}-2-grams': norm_n_sort(ngram_dict[2]), 
             f'{PREFIX}-3-grams': norm_n_sort(ngram_dict[3]), 
             f'{PREFIX}-4-grams': norm_n_sort(ngram_dict[4]), 
             f'{PREFIX}-ngram-lens': ngramlens,
             f'{PREFIX}-en-mr-ratios': ratio_dict}

for name, word_set in file_dict.items():
    with open(f'{name}.tsv', 'w', encoding='utf-8') as f:
        for key, count in word_set.items():
            f.write(f'{key}\t{count}\n')

en_types = len(en_type_dict)
with open(f'{PREFIX}-stats.txt', 'w', encoding='utf-8') as s:
    s.write(f'word tokens in total: {word_tokens}\n')
    s.write(f'english tokens in total: {en_tokens}\n')
    s.write(f'english types in total: {en_types}\n')
    s.write(f'english types to tokens ratio: {en_types/en_tokens}\n')
    s.write(f'english n-gram lenth average: {sum(lenlist)/len(lenlist)}\n')
    s.write(f'english n-gram lenth median: {statistics.median(lenlist)}\n')
    s.write(f'en/mr ratio average: {sum(en_ratios)/len(en_ratios)}\n')
    s.write(f'en/mr ratio median: {statistics.median(en_ratios)}\n')
    
print('\rruntime:', round((time.time()-start)/60,2), 'minutes')











# notes
# 7 (['only', 'assemb'], {'fft': 0.4117129214137094, 'zipf': 3.06, 'lagid>0': False, 'scowl%': 0.5})
# (['Now'], 0.6306376878055744) ???
# spelling errors
# 39 (['cricus'], {'fft': 0.7118157327495283, 'zipf': 0.0, 'lagid>0': True, 'scowl%': 0.0})

# Okok generate a full type frequency list for english and marathi tagged words and use Fitz to approximate error? Because it's proportional. But mm would have to find the parameters
