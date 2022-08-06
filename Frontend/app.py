from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.debug import console

from nltk import word_tokenize
from nltk.stem import PorterStemmer
import pickle
import re
import langid
from sinling import SinhalaTokenizer, POSTagger

app = Flask(__name__)
app.secret_key = "question_answer"  # secret key for session


# Route for the home page
@app.route('/')
def home():  # put application's code here
    return render_template("home.html")


# Route for the generate answer page
@app.route('/generate_answer', methods=["POST", "GET"])
def generate_answer():
    if request.method == "POST":  # this will check if the request has a HTTP "POST" method

        question = request.form['question']  # this will get the value entered in the form
        session['question'] = question  # Assigning the question to a session variable
        return redirect(url_for('getanswer'))  # this will pass the question to the get answer method

    else:
        return render_template('generate_answer.html')


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


# Route for the answers page
@app.route('/answer', methods=["POST", "GET"])
def getanswer():
    question = session['question']

    print('Question : ', question)

    # If the language is Sinhala the classifier would give the output as ('si', )
    print('Language - ', langid.classify(question)[0])
    # This would check if the entered question is in the Sinhala language
    if langid.classify(question)[0] == 'si':

        # Model Code

        filename = 'Model2.sav'  # Model file

        model = pickle.load(open(filename, 'rb'))  # loading the model
        category = model.predict([question])[0]  # Predicting the category of the question
        print("Question Category : ", category)

        # Tokenization with sinling SinhalaTokenizer (''https://github.com/ysenarath/sinling)
        tokenizer = SinhalaTokenizer()
        tokenized_sentences = [tokenizer.tokenize(f'{ss}.') for ss in tokenizer.split_sentences(question)]

        # Part of speach tagging
        tagger = POSTagger()
        pos_tags = tagger.predict(tokenized_sentences)

        # Identifying the numbers in the question and assigning them into the variable using chunking
        keyword_numbers = [(word, tag) for word, tag in pos_tags[0] if (tag == 'NUM')]

        print('Keyword Numbers : ', keyword_numbers)

        # Assigning the numerical values to its variables
        num_1 = keyword_numbers[0][0]
        num_2 = keyword_numbers[1][0]

        # Converting the string to integer
        number_1 = int(num_1)
        number_2 = int(num_2)

        print("num 1 : ", num_1, " num 2 : ", num_2)

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

        print('Answer : ', answer)
        return render_template("Answer.html", question=question, category=category, answer=answer)
    else:
        return render_template("try_again.html")


# Route for the try_again page
@app.route('/try_again')
def try_again():
    return render_template("try_again.html")


# Route for the help page
@app.route('/mathq_help')
def mathq_help():
    return render_template("help.html")


if __name__ == '__main__':
    app.run()
