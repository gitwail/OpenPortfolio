from flask import Flask,redirect,url_for,request,render_template


app = Flask(__name__)



@app.route('/win/<int:score>')
def win(score):
      return f'<h1> The winner score is {score} yaaay </h1>'

@app.route('/lose/<int:score>')
def lose(score):
      return f'The loser score is {score} booouuuhh'


@app.route('/result/<int:mark>')
def result(mark):
      if mark>50:
            return redirect(url_for("win",score=mark))
      else:
            return redirect(url_for("lose",score=mark))


@app.route('/user/',methods=['POST'])
def display_usr():
      return  request.json['name']+"bani"


@app.route('/',methods=['GET','POST'])
def login():
      if request.method=="GET":
            return render_template("form.html")
      else:
            name=request.form['nm']
            return render_template("hello.html",name=name)










if __name__ == '__main__':
	app.run(host='0.0.0.0', port=5000, debug=True)