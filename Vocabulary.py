from google.cloud import bigquery
import re
import os
#import sqlite3


class Vocabulary(object):
    PAD = '<PAD>'  # INDEX: 0
    SOS = '<SOS>'  # INDEX: 1
    EOS = '<EOS>'  # INDEX: 2
    UNK = '<UNK>'  # INDEX: 3
    special_tokens = [PAD, SOS, EOS, UNK]

    def __init__(self, max_len=150, dictionary_size=None):
        self.max_len = max_len
        self.word2idx = {}
        self.idx2word = {}
        self.word_occurrence = {}
        self.current_index = 4
        self.dict_size = dictionary_size

        self.inp = []
        self.tar = []
        self.tokenized = False

        for nr, word in enumerate(Vocabulary.special_tokens):
            self.word2idx[word] = nr
            # self.word_occurrence[word] = 9999999999
            self.idx2word[nr] = word

# -------------------MAIN-FUNCTIONS-------------------------

    def create_index(self, text, creating_indices=True):
        if not isinstance(text, str):
            text = ''.join(map(str, text))

        text = text.strip()
        for word in text.split(' '):
            if word in self.word_occurrence:
                self.word_occurrence[word] += 1
            else:
                self.word_occurrence[word] = 1

        if self.dict_size != None:
            self.sort_by_occurence()
            self.remove_words()

        if creating_indices:
            for word in self.word_occurrence:
                self.word2idx[word] = self.current_index
                self.idx2word[self.current_index] = word
                self.current_index += 1

    def add_words(self, text):
        for word in text:
            if word in self.word_occurrence:
                self.word2idx[word] = self.current_index
                self.idx2word[self.current_index] = word
                self.current_index += 1

    def add_words_aux(self, text, aux_vocab):
        for word in text.split(' '):
            if word not in aux_vocab:
                aux_vocab.append(word)
                self.word2idx[word] = self.current_index
                self.idx2word[self.current_index] = word
                self.current_index += 1

        return aux_vocab

    def remove_words(self):
        i = 0
        rem = []
        for word in self.word_occurrence:
            i += 1
            if i > self.dict_size:
                rem.append(word)

        for itm in rem:
            del self.word_occurrence[itm]

    def size(self):
        return len(self.word_occurrence)

    def word_exists(self, word):
        return word in self.word_occurrence

    def sort_by_occurence(self):
        self.word_occurrence = {k: v for k, v in sorted(
            self.word_occurrence.items(), key=lambda item: item[1], reverse=True)}

    def decode_text(self, enc_text, remove_borders=False):
        dec_text = []
        for idx in enc_text:
            # dec_text.append(self.idx2word[idx])

            if remove_borders:
                if idx == 1 and idx == 2:
                    continue

            if idx in self.idx2word and idx != 0:
                #dec_text += self.idx2word[idx] + ' '
                dec_text.append(self.idx2word[idx])

        return dec_text

    def encode_text(self, dec_text):
        if not isinstance(dec_text, str):
            dec_text = ''.join(map(str, dec_text))

        #dec_text = Vocabulary.punctuate_text(dec_text)
        #dec_text = Vocabulary.normalize_text(dec_text)
        dec_text = self.integrate_special_tokens(dec_text)

        enc_text = []
        for word in dec_text.split(' '):
            if word in self.word_occurrence:
                enc_text.append(self.word2idx[word])
            elif word in Vocabulary.special_tokens:
                enc_text.append(Vocabulary.special_tokens.index(word))

        return enc_text

    def pad_text(self, enc_text):
        while len(enc_text) < self.max_len:
            enc_text.append(0)  # PAD
        return enc_text

    def get_final_text(self, enc_text):
        if len(enc_text) > self.max_len:
            enc_text = enc_text[:self.max_len - 1]
            enc_text.append(2)  # EOS
        elif len(enc_text) < self.max_len:
            enc_text = self.pad_text(enc_text)

        return enc_text

    def preproc(self, text):
        text = Vocabulary.punctuate_text(text)
        text = Vocabulary.normalize_text(text)
        text = Vocabulary.normalize_numbers(text)
        text = self.encode_text(text)
        text = self.get_final_text(text)
        return text

    def tokenize_data(self):
        if self.tokenized == False:
            i = 0
            for _ in self.inp:
                enc_text = self.encode_text(self.inp[i])
                self.inp[i] = self.get_final_text(enc_text)
                enc_text = self.encode_text(self.tar[i])
                self.tar[i] = self.get_final_text(enc_text)
                i += 1
            self.tokenized = True

    def de_tokenize_data(self):
        if self.tokenized == True:
            i = 0
            for _ in self.inp:
                self.inp[i] = self.decode_text(self.inp[i])
                self.tar[i] = self.decode_text(self.tar[i])

                j = 0
                while j < len(self.inp[i]):
                    if self.inp[i][j] in Vocabulary.special_tokens[:-1]:
                        del self.inp[i][j]
                        j -= 1
                    j += 1

                j = 0
                while j < len(self.tar[i]):
                    if self.tar[i][j] in Vocabulary.special_tokens[:-1]:
                        del self.tar[i][j]
                        j -= 1
                    j += 1
                i += 1
            self.tokenized = False

    def print_data(self):
        print(' >>> VOCAB_SIZE: {}\n >>> CURENT INDEX: {}'.format(
            self.dict_size, self.current_index))

# ----------------SAVING/LOADING-DATA---------------------

    def save_csv(self, path, save_index=True):
        total_words = self.size()
        with open(path, 'w', encoding='utf-8') as f:
            if save_index:
                f.write('word,index,occurrence\n')
                for i in range(4, total_words):
                    word = self.idx2word[i]
                    occurrence = self.word_occurrence[word]
                    f.write('{},{},{}\n'.format(word, i, occurrence))
            else:
                f.write('word,occurrence\n')
                for word in self.word_occurrence:
                    f.write('{},{}\n'.format(word, self.word_occurrence[word]))

    @staticmethod
    def load_csv(path, word_occ_only=False, limit=None, verbose=True):
        v = Vocabulary()
        with open(path, 'r', encoding='utf-8') as f:
            i = 1
            for line in f.readlines()[1:]:
                text = line.split(',')
                if word_occ_only == False:
                    v.word2idx[text[0]] = text[1]
                    v.idx2word[text[1]] = text[0]
                    v.word_occurrence[text[0]] = int(text[2][:-1])
                else:
                    v.word_occurrence[text[0]] = int(text[1][:-1])

                if limit != None and i > limit:
                    break
                if verbose and i % 200000 == 0:
                    print('{} rows added to vocabulary!'.format(i))
                i += 1

        sz = v.size()
        v.current_index = sz + 1
        # vocabulary.dict_size = sz
        return v

    @staticmethod
    def create_query(sql, credentials):
        try:
            client = bigquery.Client(
                credentials=credentials, project=credentials.project_id,)

            return client.query(sql).result()

        except Exception as e:
            print(e)

    def load_bigquery_main_vocab(self, credentials, limit=None, verbose=True):
        """ Used only for creating intitial vocabulary """
        query = 'SELECT parent, comment FROM `reddit-chatobot.Reddit_db.mainTable`'
        if limit != None:
            query += ' LIMIT {}'.format(limit)

        i = 1
        rows = Vocabulary.create_query(query, credentials)
        for row in rows:
            parent = Vocabulary.punctuate_text(str(row.parent))
            parent = Vocabulary.normalize_text(parent)
            self.create_index(parent, creating_indices=False)
            comment = Vocabulary.punctuate_text(str(row.comment))
            comment = Vocabulary.normalize_text(comment)
            self.create_index(comment, creating_indices=False)

            if verbose and i % 200000 == 0:
                print('   >>> {} rows done!'.format(i))
            i += 1

        sz = self.size()
        self.current_index = sz + 1
        # vocabulary.dict_size = sz
        if verbose:
            self.print_data()

    def load_bigquery_main(self, credentials, limit=None, verbose=True):
        query = 'SELECT parent, comment FROM `reddit-chatobot.Reddit_db.mainTable`'
        if limit != None:
            query += ' LIMIT {}'.format(limit)

        i = 1
        rows = Vocabulary.create_query(query, credentials)
        aux_vocab = []
        for row in rows:
            parent = Vocabulary.punctuate_text(str(row.parent))
            parent = Vocabulary.normalize_text(parent)
            parent = Vocabulary.normalize_numbers(parent)
            aux_vocab = self.add_words_aux(parent, aux_vocab)

            comment = Vocabulary.punctuate_text(str(row.comment))
            comment = Vocabulary.normalize_text(comment)
            comment = Vocabulary.normalize_numbers(comment)
            aux_vocab = self.add_words_aux(comment, aux_vocab)

            if verbose and i % 200000 == 0:
                print('   >>> {} rows done!'.format(i))
            i += 1

        del aux_vocab

    def load_bigquery_vocab(self, credentials, vocab='no_ap', limit=None, verbose=True):
        if vocab == 'no_ap':
            query = 'SELECT word, occurrence FROM `reddit-chatobot.Reddit_db.lim_vocab_no_ap`'
        elif vocab == 'ap':
            query = 'SELECT word, occurrence FROM `reddit-chatobot.Reddit_db.lim_vocab`'
        else:
            raise Exception('Unknown vocab argument! Use \'no_ap\' or \'ap\'!')

        if limit != None:
            query += ' LIMIT {}'.format(limit)

        i = 1
        rows = Vocabulary.create_query(query, credentials)
        try:
            for row in rows:
                word = str(row.word)
                occ = int(row.occurrence)
                self.word_occurrence[word] = occ

                if verbose and i % 200000 == 0:
                    print('   >>> {} rows done!'.format(i))
                i += 1

            if verbose:
                print('Word occurrence created!')
        except Exception as e:
            print(e)

    def load_bigquery_vocab_from_indexed(self, credentials, vocab='no_ap', limit=None, verbose=True):
        '''
        if vocab == 'no_ap':
            query = 'SELECT * FROM `reddit-chatobot.Reddit_db.vocab_no_ap_indexed`'
        elif vocab == 'ap':
            query = 'SELECT * FROM `reddit-chatobot.Reddit_db.vocab_ap_indexed`'
        else:
            raise Exception('Unknown vocab argument! Use \'no_ap\' or \'ap\'!')
        '''
        query = 'SELECT * FROM `reddit-chatobot.Reddit_dbV2.full_vocabulary_validated`'

        if limit != None:
            query += ' LIMIT {}'.format(limit)

        i = 1
        rows = Vocabulary.create_query(query, credentials)
        try:
            for row in rows:
                word = str(row.word)
                occ = int(row.occurrence)
                idx = int(row.idx)
                self.word_occurrence[word] = occ
                self.word2idx[word] = idx
                self.idx2word[idx] = word

                if verbose and i % 200000 == 0:
                    print('   >>> {} rows done!'.format(i))
                i += 1

            if verbose:
                print('Word occurrence created!')
        except Exception as e:
            print(e)

    @staticmethod
    def load_bigquery_full(credentials, max_len=150, vocab='no_ap', limit_main=None, limit_vocab=None, verbose=True):
        v = Vocabulary(max_len=max_len)
        v.load_bigquery_main(credentials, limit_main, verbose)
        v.load_bigquery_vocab(credentials, vocab, limit_vocab, verbose)
        v.dict_size = v.size()

        if verbose:
            v.print_data()
        return v

    @staticmethod
    def create_inputs(credentials, max_len=150, vocab='no_ap', limit_main=None, limit_vocab=None, verbose=True):
        v = Vocabulary(max_len=max_len)
        v.load_bigquery_vocab(credentials, vocab, limit_vocab, verbose)

        query = 'SELECT * FROM `reddit-chatobot.Reddit_db.inputs`'
        if limit_main != None:
            query += ' LIMIT {}'.format(limit_main)

        i = 1
        text = set()
        rows = Vocabulary.create_query(query, credentials)
        for row in rows:
            parent = str(row.parent)
            v.inp.append(parent)
            for word in parent.split(' '):
                text.add(word)

            comment = str(row.comment)
            v.tar.append(comment)
            for word in comment.split(' '):
                text.add(word)

            if verbose and i % 1000000 == 0:
                # os.system('clear')
                print('   >>> Main: {} rows done!'.format(i))

            i += 1

        v.add_words(text)
        del text

        if verbose:
            print('Main Loaded!')
        return v

    @staticmethod
    def create_inputs_from_indexed(credentials, max_len=150, vocab='no_ap', limit_main=None, limit_vocab=None, verbose=True):
        v = Vocabulary(max_len=max_len)
        v.load_bigquery_vocab_from_indexed(
            credentials, vocab, limit_vocab, verbose)

        #query = 'SELECT * FROM `reddit-chatobot.Reddit_db.inputs` WHERE LENGTH(comment)>25 AND LENGTH(parent)>25'
        query = 'SELECT * FROM `reddit-chatobot.Reddit_db.inputs`'
        if limit_main != None:
            query += ' LIMIT {}'.format(limit_main)

        i = 1
        rows = Vocabulary.create_query(query, credentials)
        for row in rows:
            parent = str(row.parent)
            v.inp.append(parent)

            comment = str(row.comment)
            v.tar.append(comment)

            if verbose and i % 1000000 == 0:
                # os.system('clear')
                print('   >>> Main: {} rows done!'.format(i))

            i += 1

        v.current_index = i

        if verbose:
            print('Main Loaded!')
        return v

    def load_vocab_from_local(self, c, limit=None, verbose=True):
        #query = 'SELECT * FROM vocabulary_no_ap_indexed ORDER BY occurrence DESC'
        query = 'SELECT * FROM full_vocabulary_validated ORDER BY occurrence DESC'
        if limit != None:
            query += ' LIMIT {}'.format(limit)

        c.execute(query)
        i = 1
        rows = c.fetchall()
        try:
            for row in rows:
                word = str(row[0])
                occ = int(row[1])
                idx = int(row[2])
                self.word_occurrence[word] = occ
                self.word2idx[word] = idx
                self.idx2word[idx] = word

                if verbose and i % 200000 == 0:
                    print('   >>> {} rows done!'.format(i))
                i += 1

            if verbose:
                print('Word occurrence created!')
        except Exception as e:
            print(e)

# ---------------------PREPROCESSING------------------------------

    def integrate_special_tokens(self, sentence):
        #sentence = Vocabulary.punctuate_text(sentence)
        #sentence = Vocabulary.normalize_text(sentence)
        sentence = list(sentence.split(' '))
        for i, word in enumerate(sentence):
            if not self.word_exists(word):
                sentence[i] = Vocabulary.UNK

        sentence.insert(0, Vocabulary.SOS)
        sentence.append(Vocabulary.EOS)
        return ' '.join(sentence)

    @staticmethod
    def normalize_numbers(text):
        text = re.sub(r'[0]', ' 0 ', text)
        text = re.sub(r'[1]', ' 1 ', text)
        text = re.sub(r'[2]', ' 2 ', text)
        text = re.sub(r'[3]', ' 3 ', text)
        text = re.sub(r'[4]', ' 4 ', text)
        text = re.sub(r'[5]', ' 5 ', text)
        text = re.sub(r'[6]', ' 6 ', text)
        text = re.sub(r'[7]', ' 7 ', text)
        text = re.sub(r'[8]', ' 8 ', text)
        text = re.sub(r'[9]', ' 9 ', text)
        return text

    @staticmethod
    def normalize_text(text):
        text = text.lower()
        text = re.sub(r"[’`]", "'", text)
        text = re.sub(r"i'm", "i am", text)
        text = re.sub(r"he's", "he is", text)
        text = re.sub(r"she's", "she is", text)
        text = re.sub(r"that's", "that is", text)
        text = re.sub(r"there's", "there is", text)
        text = re.sub(r"what's", "what is", text)
        text = re.sub(r"where's", "where is", text)
        text = re.sub(r"who's", "who is", text)
        text = re.sub(r"how's", "how is", text)
        # text = re.sub(r"it's", "it is", text)           #<--- exception
        text = re.sub(r"let's", "let us", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"shan't", "shall not", text)
        text = re.sub(r"can't", "can not", text)
        text = re.sub(r"cannot", "can not", text)
        text = re.sub(r"n't", " not", text)

        text = re.sub(r"[@\\/|~_&#=+`$*,^]", "", text)
        text = re.sub(r"[.]+", " . ", text)
        text = re.sub(r"[!]+", " ! ", text)
        text = re.sub(r"[?]+", " ? ", text)
        text = re.sub(r"[-]+", " ", text)
        text = re.sub(r"[(<{\[]", " ( ", text)
        text = re.sub(r"[)>}\]]", " ) ", text)
        text = re.sub(r'["“”]', ' " ', text)
        text = re.sub(r"[:]", " : ", text)
        text = re.sub(r"[;]", " ; ", text)
        text = re.sub(r"[%]", " % ", text)
        text = re.sub(r"[\t\r]+", " ", text)
        text = re.sub(r"[\n\t\r]", "", text)
        text = re.sub(r" +", " ", text).strip()
        return text

    @staticmethod
    def punctuate_text(text):
        text = text.strip()
        if not (text.endswith(".") or text.endswith("?") or text.endswith("!")):
            tmp = re.sub(r"'", '"', text.lower())
            if (tmp.startswith("who") or tmp.startswith("what") or tmp.startswith("when") or
                    tmp.startswith("where") or tmp.startswith("why") or tmp.startswith("how") or
                    tmp.endswith("who") or tmp.endswith("what") or tmp.endswith("when") or
                    tmp.endswith("where") or tmp.endswith("why") or tmp.endswith("how") or
                    tmp.startswith("are") or tmp.startswith("will") or tmp.startswith("wont") or tmp.startswith("can")):
                text = "{} ? ".format(text)
            else:
                text = "{} . ".format(text)
        return text

    @staticmethod
    def restore_text(text, rm_initial_tokens=True):
        if rm_initial_tokens:
            text = text[6:-6]

        text = text.replace('<UNK>', '')
        if text[0] == ' ':
            text = text[1:]
        result = ''
        for word in text.split(' '):
            if word in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] and result[-1] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                result += word
                continue
            result += ' ' + word

        result = result[1:]
        first = result[0].upper()
        result = result[1:]
        result = first + result

        result = result.replace(' .', '.')
        result = result.replace(' !', '!')
        result = result.replace(' ?', '?')
        result = result.replace(' %', '%')
        result = result.replace(' <APOSTROPHE> ', "'")
        result = result.replace(' <QUOTE> ', '"')
        result = result.replace(' i ', ' I ')
        return result
