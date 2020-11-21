#https://spjai.com/neural-network-parameter/#1_hidden_layer_sizes`
import util
import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import os.path

from sklearn.neural_network import MLPClassifier  # アルゴリズムとしてmlpを使用

def train():
    classifier = MyMLPClassifier()
    classifier.train('input_for_train.txt')

def predict(test_text):
    classifier = MyMLPClassifier()
    classifier.load_model()
    result = classifier.predict(test_text)
    return result



class MyMLPClassifier():
    model = None
    model_name = "mlp"

    def load_model(self):
        if os.path.exists(self.get_model_path())==False:
            raise Exception('no model file found!')
        self.model = joblib.load(self.get_model_path())
        self.classes =  joblib.load(self.get_model_path('class')).tolist()
        self.vectorizer = joblib.load(self.get_model_path('vect'))
        self.le = joblib.load(self.get_model_path('le'))

    def get_model_path(self,type='model'):
        return 'models/'+self.model_name+"_"+type+'.pkl'

    def get_vector(self,text):
        return self.vectorizer.transform([text])



    def train(self, csvfile):
        df = pd.read_table(csvfile,names=('category','text'))
        X, vectorizer = util.get_vector_by_text_list(df["text"])

        # loading labels
        le = LabelEncoder()
        le.fit(df['category'])
        Y = le.transform(df['category'])

        model = MLPClassifier(max_iter=500, hidden_layer_sizes=(100,),verbose=10,)
        model.fit(X, Y)

        # save models
        joblib.dump(model, self.get_model_path())
        joblib.dump(le.classes_, self.get_model_path("class"))
        joblib.dump(vectorizer, self.get_model_path("vect"))
        joblib.dump(le, self.get_model_path("le"))

        self.model = model
        self.classes = le.classes_.tolist()
        self.vectorizer = vectorizer

    def predict(self,query):
        X = self.vectorizer.transform([query])
        key = self.model.predict(X)
        #1個予測
        #return self.classes[key[0]]
        #複数予測
        pred_cate_list = pd.DataFrame(self.model.predict_proba(X)).T.sort_values(0, ascending = False).head(10).index.tolist()
        pred_val_list = pd.DataFrame(self.model.predict_proba(X)).T.sort_values(0, ascending = False).head(10).values.tolist()

        ret = []
        for i in range(len(pred_cate_list)) :
            wkret = []
            wkret.append(self.classes[pred_cate_list[i]])
            wkret.append(pred_val_list[i][0])
            ret.append(wkret)

        return ret

    def cross_validation(self,csvfile):
        self.model = MLPClassifier(max_iter=300, hidden_layer_sizes=(100,100),verbose=True,)
        df = pd.read_csv(csvfile,names=('text','category'))
        _items = df["text"]
        X, vectorizer = nlp_tasks.get_vector_by_text_list(_items)

        # loading labels
        le = LabelEncoder()
        le.fit(df['category'])
        Y = le.transform(df['category'])
        scores = cross_val_score(self.model, X, Y, cv=4)
        print(scores)
        print(np.average(scores))




if __name__ == '__main__':

    '''
    #トレーニング
    print('学習開始：' + str(datetime.datetime.now()))
    train()
    print('学習完了：' + str(datetime.datetime.now()))
    '''

    #クロスバリデーション
    #classifier = MyMLPClassifier()
    #classifier.cross_validation('corpus.csv')

    #予測
    df_pred = pd.read_table('input_for_predict.txt',names=('category','text'))
    out_file = open('predict_result.txt','w', encoding='utf-8')

    for t in df_pred.values.tolist() :
        predict_result = predict(t[1])
        if t[0] in list(zip(*predict_result))[0] :
            comment = '    OK'
        else :
            comment = '    NG'
        print("真値：" + str(t[0]) + comment )
        wkwrite = comment + '\t' + str(t[0]) + '\t' + str(t[1])

        i = 1
        for x in predict_result :
            print("   予測値" + str(i) + "：" + str(x[0]) + "     スコア：" + str(x[1]))
            wkwrite = wkwrite + '\t' + str(x[0]) + '\t' + str(x[1])
            i += 1

        print(t[1])
        out_file.write(wkwrite + '\n')
        print("--------------------------------------")

    out_file.close()
