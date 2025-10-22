from collections import defaultdict
import time
import nltk
import spacy
spacy_nlp = spacy.load('en_core_web_sm')
from spacy.tokens import Doc
import stanza
stanza_nlp = stanza.Pipeline(lang='en', processors='tokenize,pos', tokenize_pretokenized=True, verbose=False) # lemma is slower wtih gpu... hmmm, run lemmas separately on CPU?


'''
Universal POS tagset used by SpaCy and Stanza:
ADJ   adjective
ADP   adposition (preposition/postposition)
ADV   adverb
AUX   auxiliary verb (is, have, will)
CCONJ coordinating conjunction (and, but, or)
DET   determiner (the, this)
INTJ  interjection (oh, wow)
NOUN  common noun (dog, car)
NUM   numeral (one, two)
PART  particle (to in “to go”, not in “do not”)
PRON  pronoun (he, she, it)
PROPN proper noun (London, Alice)
PUNCT punctuation
SCONJ subordinating conjunction (because, although)
SYM   symbol (+, %)
VERB  main verb (run, eat)
X     other / unclassified (foreign word, typo)
'''

# Mapping the Penn Treebank tagset used by nltk to UPOS
Penn_to_upos = {
    'CC': 'CCONJ', #   Coordinating conjunction (and, but, or)
    'CD': 'NUM', #   Cardinal number (one, two, 2, 2020)
    'DT': 'DET', #   Determiner (a, an, the, that)
    'EX': 'PRON', #   Existential there (there is)
    'FW': 'X', #   Foreign word
    'IN': 'ADP', #   Preposition or subordinating conjunction (in, on, although)
    'JJ': 'ADJ', #   Adjective (green, quick)
    'JJR': 'ADJ', #  Adjective, comparative (faster)
    'JJS': 'ADJ', #  Adjective, superlative (fastest)
    'LS': 'X', #   List item marker (1., A.)
    'MD': 'VERB', #   Modal (can, should, will)
    'NN': 'NOUN', #   Noun, singular or mass (dog, water)
    'NNS': 'NOUN', #  Noun, plural (dogs)
    'NNP': 'PROPN', #  Proper noun, singular (London, John)
    'NNPS': 'PROPN', # Proper noun, plural (Americans)
    'PDT': 'DET', #  Predeterminer (all, both, half)
    'POS': 'PART', #  Possessive ending ('s)
    'PRP': 'PRON', #  Personal pronoun (I, you, he)
    'PRP$': 'PRON', # Possessive pronoun (my, your, his)
    'RB': 'ADV', #   Adverb (quickly, silently)
    'RBR': 'ADV', #  Adverb, comparative (faster)
    'RBS': 'ADV', #  Adverb, superlative (fastest)
    'RP': 'PART', #   Particle (up, off, out)
    'SYM': 'X', #  Symbol (+, %, &)
    'TO': 'PART', #   'to' as preposition or infinitive marker
    'UH': 'INTJ', #   Interjection (uh, oh, oops)
    'VB': 'VERB', #   Verb, base form (run)
    'VBD': 'VERB', #  Verb, past tense (ran)
    'VBG': 'VERB', #  Verb, gerund or present participle (running)
    'VBN': 'VERB', #  Verb, past participle (run, eaten)
    'VBP': 'VERB', #  Verb, non-3rd person singular present (run)
    'VBZ': 'VERB', #  Verb, 3rd person singular present (runs)
    'WDT': 'DET', #  Wh-determiner (which, that)
    'WP': 'PRON', #   Wh-pronoun (who, what)
    'WP$': 'PRON', # Possessive wh-pronoun (whose)
    'WRB': 'ADV', #  Wh-adverb (where, when, how)
}

#######################################################################################################################
#################################################### POS TAGGING ######################################################
#######################################################################################################################

#nltk - accuracy: moderate
#spaCy - accuracy: high
#Stanza - accuracy: high


#nltk.download('words')
#nltk.download('punkt_tab')
#nltk.download('averaged_perceptron_tagger_eng')

def nlkt_tagger(token):
    pos_tuple = nltk.pos_tag([token])
    return Penn_to_upos[pos_tuple[0][1]]

def spacy_tagger(token):
    doc = spacy_nlp(token)
    return doc[0].pos_

def tag_tokens(token):
    doc = stanza_nlp(token)
    for sentence in doc.sentences:
        for word in sentence.words:
            return word.upos

# ^ ^ ^ Slow single-token taggers ^ ^ ^ #
#########################################
# v v v Fast multi-token taggers v v v  #

def choose_tag(a, b, c):
    if a == b or a == c:
        return a
    if b == c:
        return b
    # all distinct
    return b

def tag_tokens(tokens):
    # NLTK
    nltk_tags = [Penn_to_upos[tag] for (word, tag) in [nltk.pos_tag([token])[0] for token in tokens]]

    # spaCy
    docs = [Doc(spacy_nlp.vocab, words=[t]) for t in tokens]
    for _, proc in spacy_nlp.pipeline:
        docs = [proc(doc) for doc in docs]
    spacy_tags = [doc[0].pos_ for doc in docs]

    # Stanza
    stanza_doc = stanza_nlp([[t] for t in tokens])
    stanza_tags = [w.upos for s in stanza_doc.sentences for w in s.words]

    # Combine
    final_tags = ["INTJ" if tok == "thanks" else choose_tag(a, b, c) for tok, a, b, c in zip(tokens, nltk_tags, spacy_tags, stanza_tags)]
    return final_tags


test = ['run', 'fast', 'hello', 'world', 'serious']


#quit()
#######################################################################################################################
######################################################## MAIN #########################################################
#######################################################################################################################

start = time.time()

with open('Full-run/5M-en-types.tsv', 'r', encoding='utf-8') as f: 
    en_types = [line.split('\t') for line in f.read().splitlines()]

combined_data = []
unscaled_freqs = defaultdict(int)
scaled_freqs = defaultdict(int)
most_common = defaultdict(dict)

tokens = [t for t, _ in en_types]


start = time.time() # 2.9s/1k tokens -- 20k tokens in total -- Full run: 1 minute
token_tags = tag_tokens(tokens)
print(time.time()-start)

for (token, freq), tag in zip(en_types, token_tags):
    unscaled_freqs[tag] += 1
    scaled_freqs[tag] += int(freq)
    most_common[tag][token] = int(freq)

print(most_common['X'])

unscaled_freqs = dict(sorted(unscaled_freqs.items(), key=lambda x: x[1], reverse=True))
scaled_freqs = dict(sorted(scaled_freqs.items(), key=lambda x: x[1], reverse=True))


for tag in most_common:
    with open(f'POS-tags/{tag}s.txt', 'w', encoding='utf-8') as f:
        for word, freq in most_common[tag].items():
            f.write(f'{word}\t{str(freq)}\n')

files = {
    'absolute': unscaled_freqs,
    'relative': scaled_freqs
}

for name, freqs in files.items():
    with open(f'POS-tags/{name}-POS-frequencies.txt', 'w', encoding='utf-8') as f:
        for tag, count in freqs.items():
            f.write(f'{tag}\t{count}\n')

print(time.time()-start)
