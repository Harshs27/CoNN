
def setup():
    with open('stopwords', encoding='utf-8', errors='ignore', mode='r') as myfile:
        extra_stopwords = myfile.read().replace('\n', ' ').split(' ')[:-1]
    stop = set(stopwords.words('english'))
    # update more stopwords
    stop.update(extra_stopwords)
    #stop.update(['would', 'like', 'know', 'also', 'may', 'use', 'dont', 'get', 'com', 'write', 'want', 'edu', 'articl', 'article'])
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()
    stemmer = PorterStemmer()
    return

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop and not i.isdigit()])
    new_stop_free = " ".join(re.findall(r"[\w']+", stop_free))
    punc_free = ''.join(ch for ch in new_stop_free if ch not in exclude)
    stop_punc_free = " ".join([i for i in punc_free.lower().split() if i not in stop])
    normalized = " ".join(lemma.lemmatize(word) for word in stop_punc_free.split())
    words = []
    for word in normalized.split():
        if word in ['oed', 'aing']:
            continue
        else:
            stemmed_word = stemmer.stem(word)
            if(len(stemmed_word)>2):
                words.append(stemmed_word)
    stemmed = " ".join(words)
    stemmed_stop_free = " ".join([i for i in stemmed.split() if i not in stop])
    return stemmed_stop_free

def doc2idx(dictionary, document, unknown_word_index=-1):
        if isinstance(document, string_types):
            raise TypeError("doc2idx expects an array of unicode tokens on input, not a single string")

        document = [word if isinstance(word, str) else unicode(word, 'utf-8') for word in document]
        return [dictionary.token2id.get(word, unknown_word_index) for word in document]


def prepare_data(basefile):
    setup()
    data_cleaned = [] # format: <label, text>
    count = 0
    for path in sorted(glob(basefile+'/*')):
        category_type = os.path.basename(path)#path.split("/")[-1]
        print('category_type', category_type)
        for files in sorted(glob(basefile+'/'+category_type+'/processed.review')):#contains 342k reviews
            print('files', files)
            with open(files, encoding='utf-8', errors='ignore', mode='r') as myfile:
                file_data = myfile.readlines()#.replace('\n', ' ')
            file_data = [x.strip().split(' ') for x in file_data] # converting file data to list of lines
            for line in file_data:
                sentence = []
                label = 0 if line[-1].split(':')[-1] == 'negative' else 1 # negative label=0
                #print(line[0:-1])
                line_data = [w.split(':') for w in line[0:-1]]
#                print(line_data)
                for pair in line_data:
                    w  = ' '.join(pair[0].split('_'))# word
                    wc = int(pair[1])#word count
                    for c in range(wc): # count in wc
                        sentence.append(w)
                sentence = ' '.join(sentence)
                data_cleaned.append([label, clean(sentence).split()])
                count += 1
#                print(complete_data)
    print('total review texts ', count)
    return data_cleaned
