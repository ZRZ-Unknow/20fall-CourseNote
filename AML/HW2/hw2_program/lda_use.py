import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from utils import load_data
import argparse

def main(args):
    np.random.seed(args.seed)
    data = load_data()
    tf_vec = CountVectorizer(max_df=0.95,min_df=2,max_features=1500,stop_words='english')
    tf = tf_vec.fit_transform(data)
    topic_nums = args.topic_nums
    top_k_words = args.top_k_words
    max_iter_nums = args.max_iter_nums
    lda = LatentDirichletAllocation(n_components=topic_nums,max_iter=max_iter_nums,learning_method='batch')
    lda.fit(tf)
    tf_feature = tf_vec.get_feature_names()
    with open(f'./assets/results_use_topic{topic_nums}_iter{max_iter_nums}.txt','w') as f:
        for topic_idx,topic in enumerate(lda.components_):
            f.write(f'Topic {topic_idx}:\n')
            for i in topic.argsort()[:-top_k_words-1:-1]:
                f.write(f'  {tf_feature[i]} {topic[i]/sum(topic)}\n')

if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seed',default=0,type=int)
    p.add_argument('--topic_nums',default=5,type=int)
    p.add_argument('--top_k_words',default=10,type=int)
    p.add_argument('--max_iter_nums',default=1000,type=int)
    args = p.parse_args()
    main(args)
