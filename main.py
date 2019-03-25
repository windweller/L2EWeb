from flask import Flask, request, render_template, abort, jsonify
from api import decode_sent

app = Flask(__name__)  # , static_url_path=''


# @app.route("/", methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         inp = request.form['inp']
#         oup = decode_sent(inp)
#         return render_template('main.html', oup=oup, inp=inp)
#     return render_template('main.html')

@app.route("/", methods=['GET'])
def login():
    return render_template('chat.html')


# @app.route('/<string:page_name>/')
# def render_static(page_name):
#     return render_template('%s.html' % page_name)

@app.route('/api/decode', methods=['POST'])
def decode():
    if not request.json or not 'text' in request.json:
        abort(400)
    inp = request.json['text']
    is_why = request.json['is_why']

    try:
        src, tgt, score = decode_sent(inp, is_why=is_why)

        print(src)
        print(tgt)
        print(score)
        return jsonify({'src': src, 'tgt': tgt, 'score': score}), 201
    except:
        return jsonify({'src': inp, 'tgt': "We have internal errors possibly caused by dependency parsing.", 'score': -100}), 201


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
