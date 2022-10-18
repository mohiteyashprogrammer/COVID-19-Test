
from fileinput import close
from flask import Flask, render_template, request
app = Flask(__name__)
import pickle

file =open("model.pkl", "rb")
clf = pickle.load(file)
file.close()


@app.route('/', methods=["GET, POST"])
def hello_world():
    if request.method == "POST":
        print(request.form)

    # Code for inference
    inputfeatures = [100, 1, 22 , -1, 1]
    infprob = clf.predict_proba([inputfeatures])[0][1]
    return render_template("index.html")
    #return 'Hello, World!' + str(infprob)



if __name__ == '__main__':
    app.run(debug=True)