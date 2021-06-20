import pandas as pd

# Load the datafile
df = pd.read_csv("data/Data_train_full.csv")

# Mention the number of data points are required for the program
count = 100000

# claims and classes
claims = df['claims']
classes = df['classes']

# Dataframe to List
df = df.values.tolist()
claims = claims.values.tolist()
classes = classes.values.tolist()


# Intitalizing three types of lists for three types of text file
sentences = []
positive = []
negative = []


# +1 Class label for positve 
# -1 Class lable for negative 
# 0  Class label for information not enough
for i in range(len(df)):
    if(classes[i] == 1):
        sentences.append(df[i])
        positive.append(claims[i])
    elif(classes[i] == -1):
        sentences.append(df[i])
        negative.append(claims[i])



# Making lists with the required data points
sentences = sentences[0:count]
positive = positive[0:count]
negative = negative[0:count]

# List to dataframe
sentence = pd.DataFrame(sentences)
positive = pd.DataFrame(positive)
negative = pd.DataFrame(negative)

# Exporting data frames as text files
sentence.to_csv('data.txt', index=False, header=None)
positive.to_csv('positive.txt', index=False, header=None)
negative.to_csv('negative.txt', index=False, header=None)