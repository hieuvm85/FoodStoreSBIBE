from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from PIL import Image
from io import BytesIO
from kmean import train,search
import time


app = Flask(__name__)
# CORS(app)

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:hieu08052002@localhost/kmeanfood'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False 
db = SQLAlchemy(app)

# Định nghĩa route cơ bản
@app.route('/')
def home():
    return "Chào mừng bạn đến với API FoodStoreSBIBE!"

@app.route('/train', methods=['GET'])
def getImages():
    time.sleep(5)
    train()
    return jsonify({"data": "success"}),200

@app.route('/search', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    try:
        # Mở và xử lý ảnh
        image = Image.open(BytesIO(file.read()))
        images= search(image)

        return jsonify({
            'status': 'success',
            'images': images
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run()

