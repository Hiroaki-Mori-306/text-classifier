#https://spjai.com/category-classification/
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score

def _split_to_words(text):
    doc = text.replace("\u3000","").replace("\n","")
    doc = re.sub(r'[0123456789０１２３４５６７８９！＠＃＄％＾＆\-|\\＊\“（）＿■×※⇒—●(：〜＋=)／*&^%$#@!~`){}…\[\]\"\'\”:;<>?＜＞？、。・,./『』【】「」→←○]+', "", doc)
    word_list = ''
    for x in nltk.pos_tag(doc):
        if x[1] in [
                        'NN','NNP','NNPS','NNS',
                        'VB','VBD','VBG','VBN','VBP','VBZ','',
                        'JJ','JJR','JJS',
                        'RB',
                    ]:
            word_list = word_list + x[0] + ' '
    return word_list

def get_vector_by_text_list(_items):
#    count_vect = CountVectorizer(analyzer=_split_to_words)
    count_vect = TfidfVectorizer(analyzer=_split_to_words)
    bow = count_vect.fit_transform(_items)
    X = bow.todense()
    return [X,count_vect]

if __name__ == '__main__':
    print('aaa')
