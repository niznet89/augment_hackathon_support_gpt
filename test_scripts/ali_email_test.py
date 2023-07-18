from flask import Flask, request

app = Flask(__name__)


@app.route('/email_received', methods=['POST'])
def email_received():
    # your code goes here
    print(request.json)  # for debugging purposes
    # trigger your event
    return 'Success', 200


if __name__ == '__main__':
    app.run(port=5000)
