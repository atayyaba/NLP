import pickle
import string
import nltk

from nltk import word_tokenize, re
from nltk.corpus import stopwords, wordnet
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination as VE
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd


from tkinter import * 
from tkinter import ttk


#______________________________________________________________________________
# Bayesian Network 


#_____________________________________________________________________________________________________________________

# Design Graphicall  User Interface 
#___________________________________
 
window = Tk() 
window.title("Automatic Classification of Non-Functional Requirements using Probabilistic Graphical Model") # title of the Window
window.geometry('1280x1024') # Grid size of the window


def RESULT():
    from nltk.corpus import stopwords
    import pickle
    import string
    import nltk

    from nltk import word_tokenize, re
    from nltk.corpus import stopwords, wordnet
    from pgmpy.models import BayesianModel
    from pgmpy.inference import VariableElimination as VE
    from sklearn.feature_extraction.text import CountVectorizer
    import pandas as pd


    words = []
    Input = review.get('1.0', END)
    for word in Input:
         text_tokens = word_tokenize(word)
         tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
         words.append(tokens_without_sw)
    stopwords = nltk.corpus.stopwords.words('english')  # this will remove stop words already present in nltkcorpus
    wn = nltk.WordNetLemmatizer()
    filename = 'myModel.pkl'
    listforAug = []
    AugmentView = []
    words = []

    def clean_text(text):  # this will clean the data as well as split/tokenized it and find the sysnonmn of each word
        text = "".join([word for word in text if word not in string.punctuation])
        tokens = re.split("\W+", text)
        text = [wn.lemmatize(word) for word in tokens if word not in stopwords]
        lemmas = []
        for token in text:
            lemmas += [synset.lemmas()[0].name() for synset in
                       wordnet.synsets(token)]  # this will find the sysnomum of a word
        return list(set(lemmas))
    for i in words:
         words = (" ").join(i)
         listforAug.append(words)
    for i in listforAug:
        #print("Without Augment: ", i)
        a = clean_text(i)
        a = (" ").join(a)
        AugmentView.append(str(i) + str(a))    
    d = []
    for i in AugmentView:
        d.append(i)
    list1=[Input]
    vectorizer = CountVectorizer(ngram_range=(1, 1), stop_words='english')
    X = vectorizer.fit_transform(list1)
    cv_dataframe = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())
    dictionaryObject = cv_dataframe.to_dict(orient='records')
    y=dictionaryObject[0]
    loaded_modelNN = pickle.load(open(filename, 'rb'))
    infer = VE(loaded_modelNN)
    q=infer.query(variables=['Functional'], evidence =y)
    
    reviewlist.insert('1.0',q) # Print List of All Classes 


    prob = list(q.values) # save values of all probabilities in prob ( in the form of list)
    target = max(prob)  # Get maximum probability from the list inorder to get the 1 label of the class
    index= prob.index(target) # return the index of the row whichh has highest probability
    label=['Performance Efficiency','Portability','Relaibility','Security','Usability']
    print(label[index])
    AL = q.assignment(prob) # Corresponding labels of each probability ( now AL has all labels of the q class )
    #for clas, label in AL[index]: # Iter the second element of the pair ( class , label) to get the required label
    reviewlabel.insert(0,label[index]) # Print 1 class
# =============================================================================
# GUI for Focus of Interest
#      
# =============================================================================
# For TITLE Of the Interface     
title =Label(window, text=" USER REVIEW" , font='Times 16 bold italic') 
free=Label(window)
free.grid(column=1, row=1)
title.grid(column=1, row=2)  
free=Label(window)
free.grid(column=1, row=3)

# ENTER REVIEW BY THE USER
subtitle =Label(window, text=" Enter User Review" , font='Times 14 bold') 
subtitle.grid(column=1, row=4)
review = Text(window,height=7, width=40)
review.grid(column=1, row=5)

#Find FINAL LIST oF CLASS

subtitle =Label(window, text="===========================" , font='Times 14 bold') 
subtitle.grid(column=1, row=6)  
 
ML= Label(window, text="List of Multiple Labels", font='Times 14 bold ')
ML.grid(column=1, row=7)
free=Label(window)
free.grid(column=1, row=8)
reviewlist= Text(window, height=16, width=70)
reviewlist.grid(column=1, row=9)

# FIND FINAL CLASS
style=Label(window, text="===========================" , font='Times 14 bold') 
style.grid(column=1, row=10) 
lblforjournal = Label(window, text="Final Class ", font='Times 14 bold')
lblforjournal.grid(column=1, row=11)
free=Label(window)
free.grid(column=1, row=9)
reviewlabel= Entry(window, width=30)
reviewlabel.grid(column=1, row=12)

# BUTTON 

button4 = Button(window, text="Find the Class", command= RESULT)
button4.grid(column=1, row=13)
#______________________________________________________________________________

window.mainloop()

     