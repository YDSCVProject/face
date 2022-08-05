from crypt import methods
from flask_ngrok import run_with_ngrok
from flask import Flask
from flask import render_template , url_for, redirect, flash
from flask import request
import pandas as pd
import os
from werkzeug.utils import secure_filename

# MODEL
import os
from collections import OrderedDict
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer

opt = TestOptions().parse(save=False)
opt.display_id = 0 # do not launch visdom
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.in_the_wild = True # This triggers preprocessing of in the wild images in the dataloader
opt.traverse = True # This tells the model to traverse the latent space between anchor classes
opt.interp_step = 0.05 # this controls the number of images to interpolate between anchor classes

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
visualizer = Visualizer(opt)

# !ngrok authtoken <token password>
UPLOAD_FOLDER = 'static/uploads/'
app = Flask(__name__, static_folder='static')
# run_with_ngrok(app) 

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])

def index():
    return render_template('index.html', title='face GAN')

@app.route('/index', methods=['POST'])

def aging():
    age = request.form['age']
    target_age = request.form['target_age']
    gender = request.form['gender']
    image = request.files['image']
    path = os.path.join(app.config["UPLOAD_FOLDER"], image.filename)
    image.save(path)
    
    if gender == "male":
        input = pd.DataFrame({
        'age' : [int(age)],
        'target_age' :[int(target_age)]
        })

        opt.name = 'males_model'
        model = create_model(opt)
        model.eval()

    elif gender == "female":
        input = pd.DataFrame({
        'age' : [int(age)],
        'target_age' :[int(target_age)]
        })

        opt.name = 'females_model'
        model = create_model(opt)
        model.eval()  

    name = f'./static/uploads/{image.filename}'
    data = dataset.dataset.get_item_from_path(name)
    visuals = model.inference(data)

    # Model running (images)
    os.makedirs(f'results/{image.filename}', exist_ok=True)
    out_pathi = f'./results/{image.filename}'   

    visualizer.save_images_deploy(visuals, out_pathi)
    
    # Model running (video)
    os.makedirs('static', exist_ok=True)
    out_pathv = os.path.join('static', os.path.splitext(name)[0].replace(' ', '_') + '.webm')
    visualizer.make_video(visuals, out_pathv)

    return render_template('output.html', filename=image.filename) #Output = ModelOutput)

if __name__ == "__main__":
    app.run()
# app.run(host='0.0.0.0', port=5000, debug=True)