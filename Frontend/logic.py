from nltk import word_tokenize
from nltk.stem import PorterStemmer
import pickle
import re
import langid
from sinling import SinhalaTokenizer, POSTagger
import pandas as pd
import csv
import string

# Model Code
data = pd.read_csv("Dataset/DataSetShuffled.csv")
total_questions = len(data)
print("Number of rows : ", total_questions)


# # Once the shuffling of the data set is completed this set of code should be commented out
# # Shuffling the dataset
# questiondata1 = data.sample(frac=1).reset_index(drop=True)
#
# # Saving the shuffled dataset to a nes CSV file
# questiondata1.to_csv("Dataset/DataSetShuffled.csv", index=False)


# The preprocessing method of the model
def preprocess_and_tokenize(data):
    # remove html markup
    data = re.sub("(<.*?>)", "", data)

    # remove urls
    data = re.sub(r'http\S+', '', data)

    # remove hashtags and @names
    data = re.sub(r"(#[\d\w\.]+)", '', data)
    data = re.sub(r"(@[\d\w\.]+)", '', data)

    # remove punctuation and non-ascii digits
    data = re.sub("(\\W|\\d)", " ", data)

    # remove whitespace
    data = data.strip()

    # tokenization with nltk
    data = word_tokenize(data)

    # stemming with nltk
    porter = PorterStemmer()
    stem_data = [porter.stem(word) for word in data]

    return stem_data


num_of_correct_answers = 0
with open('Dataset/DataSetShuffled.csv', encoding="utf8") as file_obj:
    heading = next(file_obj)

    reader_obj = csv.reader(file_obj)

    for row in reader_obj:
        question = row[0]

        filename = 'Model2.sav'  # Model file

        model = pickle.load(open(filename, 'rb'))  # loading the model
        category = model.predict([question])[0]  # Predicting the category of the question
        # print("Question Category : ", category)

        # question = re.sub("(<.>)", "", question)

        # Tokenization with sinling SinhalaTokenizer (''https://github.com/ysenarath/sinling)
        tokenizer = SinhalaTokenizer()
        tokenized_sentences = [tokenizer.tokenize(f'{ss}.') for ss in tokenizer.split_sentences(question)]

        tagger = POSTagger()
        pos_tags = tagger.predict(tokenized_sentences)

        # Identifying the numbers in the question and assigning them into the variable
        keyword_numbers = [(word, tag) for word, tag in pos_tags[0] if (tag == 'NUM')]

        # print('Keyword Numbers : ', keyword_numbers)

        num_1 = keyword_numbers[0][0]
        num_2 = keyword_numbers[1][0]

        number_1 = int(num_1)
        number_2 = int(num_2)

        # print("num 1 : ", num_1, " num 2 : ", num_2)

        # Calulation for Multiplication
        if category == 'Multiplication':
            answer = number_1 * number_2  # calculating the answer

        elif category == 'Division':
            if number_1 > number_2:
                answer = number_1 / number_2  # calculating the answer

        elif number_2 > number_1:
            answer = number_2 / number_1  # calculating the answer

        elif category == 'Addition':
            answer = number_1 + number_2  # calculating the answer

        elif category == 'Subtraction':
            if number_1 > number_2:
                answer = number_1 - number_2  # calculating the answer

        elif number_2 > number_1:
            answer = number_2 - number_1  # calculating the answer

        actual_answer = int(row[1])
        generated_answer = int(answer)

        print('Preticted Answer : ', generated_answer)
        print('Actual Answer : ', actual_answer)

        if generated_answer == actual_answer:
            print("hello")
            num_of_correct_answers = num_of_correct_answers + 1

print("Number of Correct Answers : ", num_of_correct_answers)
print("Total Number of Questions : ", total_questions)

system_accuracy = (num_of_correct_answers / total_questions) * 100

print("System Accuracy : ", system_accuracy)
