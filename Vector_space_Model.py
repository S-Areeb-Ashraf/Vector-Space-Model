import nltk
import re
import math
from nltk import PorterStemmer
s=PorterStemmer()

# Read Carefully

# Rules to Run the program:

# Unfortunately Your gold query data set has some minor issues in results
# Do not change the directory as it will cause problems
# After unzipping the folder first install nltk library
# Command for installation: pip install nltk
# Then run this file
# This file will make some warnings regarding txt files  (Ignore them)
# It will take some time to run as there are 448 files
# Some time it will take more time, (do not assume that program has crashed), it just need time to process
# Original files are in the folder named (Text_files)
# Updated Tokenized files will be created in the folder named (Updated_Text_Files)
# First Write the query when program asks about
# You will get the result set 



stop_words=[
    "a","is","the","of","all","and","to","can","be",
    "as","once","for","at","am","are","has","have","had",
    "up", "his","her","in","on","no","we","do"]

size=448

all_tf={};
idf={}
tf_idf={}
q_tfidf={}


# Function to remove stop words and punctuaion marks

def filter_words():
    for temp in range(1,size+1):
        u_all_line=[]
        with open(f"Text_files\{temp}.txt","r") as file:
            all_lines=file.readlines()

            for s_line in all_lines:
                s_line=s_line.casefold()
                s_line=re.sub(r"[^a-zA-Z\s]", " ", s_line)
                filter_w=[]
                words=s_line.split()
                for word in words:
                    if word not in stop_words:
                        filter_w.append(word)

                update_line = " ".join(filter_w)
                u_all_line.append(update_line + "\n")
        file.close()

        with open(f"Updated_Text_Files\{temp}.txt","w") as file:
            file.writelines(u_all_line)
        file.close()
    
    print("\n\n\t\t******** Welcome to Vector Space Model ********")
    print("\n\t\t** Tokenization Of Documents has been Completed **\n")
    return


# Funtion for steming of words in documents

def token_stem():
    stem_w=[]
    update_line=[]
    for i in range(1,size+1):
        update_line.clear()
        with open(f"Updated_Text_Files\{i}.txt","r") as file:
            my_list=file.readlines()
            for line in my_list:
                words=line.split()
                stem_w.clear()
                for s_word in words:
                    str1=s.stem(s_word)
                    stem_w.append(str1)
                update_line.append(" ".join(stem_w) + "\n")
        file.close()

        with open(f"Updated_Text_Files\{i}.txt","w") as file:
            file.writelines(update_line)
        file.close()
    return


# Function to calculate Term frequency (tf) in documents

def solve_tf():
    for i in range(1,size+1):
        tf={}
        doc_name=f"{i}"
        with open(f"Updated_Text_Files\{i}.txt", 'r') as file:
            for line in file:
                lines=line.strip().split()
                for s_word in lines:
                    if s_word not in tf:
                        tf[s_word]=0
                    tf[s_word]+=1

        all_tf[doc_name]=tf
    return all_tf


# Function to calculate IDF using document frequeency and total number of documents

def solve_idf():
    df={}
    for i in range(1,size+1):
        with open(f"Updated_Text_Files\{i}.txt", 'r') as file:
            tokens_in_doc=set()
            for line in file:
                lines=line.strip().split()
                tokens_in_doc.update(lines)

        for s_word in tokens_in_doc:
            if s_word not in df:
                df[s_word]=0
            df[s_word]+=1
    for word,freq in df.items():
        val=math.log(size/freq)
        idf[word]=val

    return idf


# Function to multiply tf and IDF and then store them into tf_idf

def solve_tf_idf():
    for doc_name, tf_dict in all_tf.items():
        doc_tfidf={}
        for s_word, tf in tf_dict.items():
            if s_word in idf:
                doc_tfidf[s_word]=tf*idf[s_word]
        tf_idf[doc_name]=doc_tfidf
    return tf_idf;



# Function for tokenizing and then calautaing its tf-idf value

def proc_query():
    print("\n\t\t**** Enter Query  ****")
    tf={}
    opt=input().strip()
    words=opt.split()
    for token in words:
        s_word=token.casefold()
        s_word=re.sub(r"[^a-zA-Z\s]", "",s_word)
        s_word=s.stem(s_word)
        tf[s_word]=tf.get(s_word,0)+1
    for token,freq in tf.items():
        if token in idf:
            q_tfidf[token]=freq*idf[token]
    scores={}
    for doc_name,doc_vec in tf_idf.items():
        score=check_similar(q_tfidf, doc_vec)
        scores[doc_name]=score
    
    alpha_val=0.001
    fil_scor={}
    for d_name,sim_score in scores.items():
        if sim_score>=alpha_val:
            fil_scor[d_name]=sim_score

    rank_docs=sorted(fil_scor.items(),key=lambda item: item[1],reverse=True)
    return rank_docs


# Funtion to calautae cosine andles betqeen words and query temrs

def check_similar(v1,v2):
    prod=0
    n1=0;n2=0
    for token in v1:
        if token in v2:
            prod+=v1[token]*v2[token]

    for val in v1.values():
        n1+=val*val
    for val in v2.values():
        n2+=val*val
    if n1==0 or n2==0:
        return 0
    return prod/(math.sqrt(n1)*math.sqrt(n2))


# From here all functions will be called

filter_words()
token_stem()

solve_tf()
solve_idf()
solve_tf_idf()

my_list=[]
my_list = proc_query()
my_list.sort(key=lambda x: int(x[0]))
doc_names=[doc[0] for doc in my_list]
doc_names.sort(key=lambda x: int(x))

print()
print("Answer of the QUery")
print()
print(",".join(doc_names))
print()
print(f"Length of the Answer {len(my_list)}")
