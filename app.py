from flask import Flask, render_template, request, send_file, redirect, url_for
import os
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

classename =  {
    0: 'Abissínio',
    1: 'Bengal',
    2: 'Birmanês',
    3: 'Bombay',
    4: 'British Shorthair',
    5: 'Egípcio',
    6: 'Maine Coon',
    7: 'Persa',
    8: 'Ragdoll',
    9: 'Russo Azul',
    10: 'Siamês',
    11: 'Sphynx',
    12: 'American Shorthair',
    13: 'Basset Hound',
    14: 'Beagle',
    15: 'Boxer',
    16: 'Chihuahua',
    17: 'Bulldog Inglês',
    18: 'Pastor Alemão',
    19: 'Dogue Alemão',
    20: 'Havanês',
    21: 'Akita Inu',
    22: 'Keeshond',
    23: 'Leonberger',
    24: 'Pinscher Miniatura',
    25: 'Terra Nova',
    26: 'Spitz Alemão Anão',
    27: 'Pug',
    28: 'São Bernardo',
    29: 'Samoieda',
    30: 'Fold Escocês',
    31: 'Shiba Inu',
    32: 'Staffordshire Bull Terrier',
    33: 'Terrier Irlandês',
    34: 'Yorkshire Terrier'
}

def get_class_name(class_number):
    return classename.get(class_number, "Classe não encontrada")

model = load_model('./classificador.h5')

app = Flask(__name__, template_folder='templates')
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'upload')

@app.route('/')
def home():
    return render_template('form.html')
    
@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    savePath = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    file.save(savePath)
    filename = secure_filename(file.filename.split('.')[0]) 
    return redirect(url_for('get_file', filename=filename))

@app.route('/get-file/<filename>')
def get_file(filename):
    file = os.path.join(UPLOAD_FOLDER, filename + '.jpg')

    img = image.load_img(file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    previsao = model.predict(img_array)
    previsao_classe = np.argmax(previsao, axis=1)
    
    nomeRaca = get_class_name(previsao_classe[0])
    
    image_url = f'/get-image/{filename}'
    return render_template('resultado.html', predicted_class=nomeRaca, image_url=image_url)

@app.route('/get-image/<filename>')
def get_image(filename):
    file = os.path.join(UPLOAD_FOLDER, filename + '.jpg')
    return send_file(file, mimetype='image/jpg')

if __name__ == '__main__':
    app.run(port=5000, host='localhost', debug = True) 