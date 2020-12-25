import numpy as np
import argparse,re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer,SnowballStemmer
import random,time
from utils import *
from bidict import bidict
import pickle

wnl = WordNetLemmatizer()

class Document(object):
    def __init__(self,strings):
        strings = re.sub(r'[{}]+'.format(string.punctuation),' ',strings)
        strings = strings.lower()
        tokens = nltk.word_tokenize(strings)
        tokens = [i for i in tokens if (not i in stopwords.words('english')) and len(i)>1 and (not i.isdigit())]
        s = [wnl.lemmatize(s) for s in tokens]
        words = set(s)
        self.words = list(words)
    
    def __len__(self):
        return len(self.words)

class DocCollection(object):
    def __init__(self):
        self.docs = []
        self.words = bidict()
        self.word_nums = 0
    
    def append(self, doc):
        self.docs.append(doc)
        for word in doc.words:
            if word not in self.words.values():
                self.words[self.word_nums] = word
                self.word_nums += 1

class LDA(DocCollection):
    def __init__(self, topic_nums, top_k_words, alpha, beta, max_iter_nums):
        super().__init__()
        self.topic_nums = topic_nums
        self.top_k_words = top_k_words
        self.alpha = alpha
        self.beta = beta
        self.max_iter_nums = max_iter_nums
    
    def train(self):
        self.doc_nums = len(self.docs)
        self.doc_topic_dist = np.zeros([self.doc_nums, self.topic_nums], dtype=np.int32)
        self.topic_word_dist = np.zeros([self.topic_nums, self.word_nums], dtype=np.int32)
        self.topic_dist = np.zeros(self.topic_nums, dtype=np.int32)
        self.curr_assign = []

        print(f'Initializing',end=' ',flush=True)
        t = time.time()
        for i_doc, doc in enumerate(self.docs):
            word_topic = []
            for i_word, word in enumerate(doc.words):
                assert word in self.words.values(), "word not in words!"
                index_w = self.words.inverse[word]
                index_t = np.random.randint(self.topic_nums)
                word_topic.append(index_t)
                self.doc_topic_dist[i_doc, index_t] += 1
                self.topic_dist[index_t] += 1
                self.topic_word_dist[index_t, index_w] += 1
            self.curr_assign.append(word_topic)
        print(f'ended in {(time.time()-t)/60.0} minutes')

        for i_iter in range(self.max_iter_nums):
            print(f'Iter {i_iter}',end=' ',flush=True)
            time1 = time.time()
            for i_doc, doc in enumerate(self.docs):
                for i_word, word in enumerate(doc.words):
                    assert word in self.words.values(), "word not in words!"
                    index_w = self.words.inverse[word]
                    curr_topic_index = self.curr_assign[i_doc][i_word]
                    self.doc_topic_dist[i_doc, curr_topic_index] -= 1
                    self.topic_dist[curr_topic_index] -= 1
                    self.topic_word_dist[curr_topic_index, index_w] -= 1
                    topic_dist = (self.topic_word_dist[:,index_w] + self.beta)*\
                        (self.doc_topic_dist[i_doc,:] + self.alpha)/(self.topic_dist + self.beta)
                    index_t = choose_from_dist(topic_dist)
                    self.curr_assign[i_doc][i_word] = index_t
                    self.doc_topic_dist[i_doc, index_t] += 1
                    self.topic_dist[index_t] += 1
                    self.topic_word_dist[index_t, index_w] += 1
                self.curr_assign.append(word_topic)
            print(f'ended in {(time.time()-time1)/60.0} minutes')

    def get_result(self):
        dist = (self.topic_word_dist+self.beta)/(self.topic_dist.reshape(-1,1)+self.word_nums*self.beta)
        with open(f'./assets/results_topic{self.topic_nums}_iter{self.max_iter_nums}.txt','w') as f:
            for i_topic in range(self.topic_nums):
                word_count = self.topic_word_dist[i_topic]
                word_index_count = []
                for i in range(self.word_nums):
                    word_index_count.append([i, word_count[i]])
                word_index_count = sorted(word_index_count, key=lambda x:x[1], reverse=True)
                f.write(f'Topic {i_topic}:\n')
                for i in range(self.top_k_words):
                    index = word_index_count[i][0]
                    f.write(f'  {self.words[index]} {dist[i_topic,index]}\n')

def run(args):
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    if args.load:
        lda = pickle.load(open('./assets/lda.b','rb'))
        lda.topic_nums = args.topic_nums
        lda.top_k_words = args.top_k_words
        lda.alpha = args.alpha
        lda.beta = args.beta
        lda.max_iter_nums = args.max_iter_nums
    else:
        lda = LDA(
            args.topic_nums,
            args.top_k_words,
            args.alpha,
            args.beta,
            args.max_iter_nums
        )
        data = load_data()
        print("Processing data...")
        for line in data:
            doc = Document(line)
            lda.append(doc)
    print(f"dict_size:{lda.word_nums},doc_num:{len(lda.docs)},topic_num:{lda.topic_nums},"
          f"iter_num:{lda.max_iter_nums},top_k_words:{lda.top_k_words},alpha:{lda.alpha},beta:{lda.beta}")
    #pickle.dump(lda,open('./assets/lda.b','wb'))
    print("Begin training")
    lda.train()
    print("Finish training")
    lda.get_result()

if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seed', default=0, type=int)
    p.add_argument('--topic_nums', default=5, type=int)
    p.add_argument('--top_k_words', default=10, type=int)
    p.add_argument('--alpha', default=0.1, type=float)
    p.add_argument('--beta', default=0.1, type=float)
    p.add_argument('--max_iter_nums', default=200, type=int)
    p.add_argument('--load',action='store_false')
    args = p.parse_args()
    t1 = time.time()
    run(args)
    print(f'Total time used:{(time.time()-t1)/60.0} minutes')

