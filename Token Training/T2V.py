import numpy as np
import argparse
from time import time
import nltk
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from nltk.tokenize import word_tokenize

class callback(CallbackAny2Vec):
    '''Callback to print loss after each epoch.'''
    def __init__(self):
        self.epoch = 0
        self.loss_to_be_subed = 0
        self.time = time()
        

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        loss_now = loss - self.loss_to_be_subed
        self.loss_to_be_subed = loss
        log_file = "training_"+str(model.vector_size)+".log" 

        if ((self.epoch+1) % 100 == 0) and (self.epoch != 0):
            t_time = time() - self.time
            with open(log_file, "a") as text_file:
                text_file.write("Loss after epoch {}: {:15.2f}, training time: {:10.4f} s \n".format(self.epoch+1, loss_now, t_time))
        self.epoch += 1

def preprocessing (file_content):
    corpus = []
    with open(file_content, 'r') as inputfile:
        for line in inputfile:
            tokens = word_tokenize(line)
            corpus.append(tokens)
    
    return corpus

def main():
    nltk.download('punkt')
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", help="Path to the imput",required=True)
    ap.add_argument("-p", "--param", help="Input number of vector size",type=int,required=True)
    
    args = vars(ap.parse_args())
    
    file_content = args["input"] if args["input"] else print("Please add the file location")
    vector_size = args["param"] if args["param"] else print("Please define the parameters")
    
    corpus = preprocessing (file_content)
    model2 = Word2Vec(corpus, min_count = 1, vector_size = vector_size,
                                             window = 2, sg = 1, workers=8, epochs=10000, compute_loss=True, callbacks=[callback()])

    
    outfile = "Attr2Vec"+str(vector_size)+".model"
    model2.save(outfile)


if __name__ == "__main__":
    main()


