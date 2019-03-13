from flask import Flask, request, render_template, abort, jsonify
from api import decode_sent

# def predict(text):
#     ret = []
#     for i in range(5):
#         ret.append({"label": '%d-%s' % (i, text)})
#     return ret


app = Flask(__name__)  # , static_url_path=''


@app.route("/", methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        inp = request.form['inp']
        oup = decode_sent(inp)
        return render_template('main.html', oup=oup, inp=inp)
    return render_template('main.html')


@app.route('/api/decode', methods=['POST'])
def decode():
    if not request.json or not 'text' in request.json:
        abort(400)
    inp = request.json['text']
    oup_html, ids, inputtext, top_level_ids, top_level_names = decode_sent(inp)

    print(ids)

    return jsonify({'output_html': oup_html, 'ids': ids, 'inputtext': inputtext,
                    "top_level_ids": top_level_ids, "top_level_names": top_level_names}), 201


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
