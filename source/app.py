import hashlib
import pickle
import random
import re
import os
from io import BytesIO
from typing import Optional
import torch.nn.functional as F
import PIL.Image
import PIL.ImageOps
import clip
import numpy as np
import torch
from flask import Flask, send_file, url_for
from flask import render_template, request, redirect

from data_utils import targetpad_resize, server_base_path, data_path, dataset_root
from utils import get_uri_to_metadata_dict

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = server_base_path / 'uploaded_files'
PORT = os.environ.get('PORT', 5555)

if torch.cuda.is_available():
    device = torch.device("cuda")
    data_type = torch.float16
else:
    device = torch.device("cpu")
    data_type = torch.float32


@app.route('/')
def choice():
    return redirect(url_for('query'))


@app.route('/favicon.ico')
def favicon():
    return url_for('static', filename='/favicon.ico')


@app.route('/noisyart')
def query():
    random_indexes = random.sample(range(len(test_hash_to_path)), k=30)
    names = np.array(test_index_hashes)[random_indexes].tolist()
    return render_template('query.html', dataset='noisyart', names=names)


@app.route('/noisyart', methods=['POST'])
def relative_caption():
    caption = request.form['custom_caption']
    return redirect(url_for('t2i_results', dataset='noisyart', caption=caption))


@app.route('/<string:dataset>/<string:old_caption>', methods=['POST'])
def custom_caption(dataset: str, old_caption: Optional[str] = None):
    caption = request.form['custom_caption']
    return redirect(url_for('t2i_results', dataset=dataset, caption=caption))


@app.route('/<string:dataset>/<string:caption>')
def t2i_results(dataset: str, caption: str):
    n_retrieved = 50

    text_inputs = clip.tokenize(caption, truncate=True).to(device)
    with torch.no_grad():
        query_features = F.normalize(clip_model.encode_text(text_inputs))

    index_features = F.normalize(global_index_features, dim=-1)

    cos_similarity = query_features @ index_features.T
    sorted_indices = torch.topk(cos_similarity, n_retrieved, largest=True).indices.cpu()
    sorted_index_names = np.array(global_index_hashes)[sorted_indices].flatten()

    return render_template('t2i_results.html', dataset=dataset, caption=caption, target_name="",
                           names=sorted_index_names[:n_retrieved])


@app.route('/<string:dataset>/artwork/<string:query_image_name>')
def i2i_results(dataset: str, query_image_name: str):
    n_retrieved = 50

    query_index = global_index_hashes.index(query_image_name)
    query_features = F.normalize(global_index_features[query_index], dim=-1)

    index_features = F.normalize(global_index_features)

    cos_similarity = query_features @ index_features.T
    sorted_indices = torch.topk(cos_similarity, n_retrieved * 3, largest=True).indices.cpu()
    sorted_index_names = np.array(global_index_hashes)[sorted_indices].flatten()

    # Get metadata informations
    metadata = uri_to_metadata[re.search('[0-9]*_(.*)/.*', global_hash_to_path[query_image_name]).group(1)]
    description = metadata['description']
    title = metadata['title']
    authors = ""
    for author in metadata['_authors']:
        authors += f"{author['name']}, "
    authors = authors.strip(", ")
    return render_template('i2i_results.html', dataset=dataset, query_image_name=query_image_name, target_name="",
                           names=sorted_index_names[:n_retrieved], description=description, title=title,
                           authors=authors)


@app.route('/get_image/<string:image_name>')
@app.route('/get_image/<string:image_name>/<int:dim>')
@app.route('/get_image/<string:image_name>/<int:dim>/<string:gt>')
@app.route('/get_image/<string:image_name>/<string:gt>')
def get_image(image_name: str, dim: Optional[int] = None, gt: Optional[str] = None):
    if image_name in test_hash_to_path:  # TODO
        image_path = dataset_root / 'noisyart_dataset' / 'test_200' / global_hash_to_path[image_name]
    elif image_name in trainval_hash_to_path:
        image_path = dataset_root / 'noisyart_dataset' / 'trainval_3120' / global_hash_to_path[image_name]
    else:
        raise ValueError()

    # if 'dim' is not None resize the image
    if dim:
        transform = targetpad_resize(1.25, int(dim), 255)
        pil_image = transform(PIL.Image.open(image_path))
    else:
        pil_image = PIL.Image.open(image_path)

    pil_image = pil_image.convert('RGB')
    # add a border to the image
    if gt == 'True':
        pil_image = PIL.ImageOps.expand(pil_image, border=5, fill='green')
    elif gt == 'False':
        pil_image = PIL.ImageOps.expand(pil_image, border=5, fill='red')
    elif gt is None:
        pil_image = PIL.ImageOps.expand(pil_image, border=1, fill='grey')

    img_io = BytesIO()
    pil_image.save(img_io, 'JPEG', quality=80)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')

@app.before_first_request
def _load_assets():
    noisyart_imgs_path = str(dataset_root / 'noisyart_dataset' / 'trainval_3120')
    noisyart_json_path = str(dataset_root / 'noisyart_dataset' / 'noisyart' / 'metadata.json')

    global uri_to_metadata
    uri_to_metadata = get_uri_to_metadata_dict(noisyart_imgs_path, noisyart_json_path)

    # Trainval ---------------------------
    global trainval_index_features
    trainval_index_features = torch.load(
        data_path / 'trainval_index_features.pt', map_location=device).type(data_type).cpu()

    global trainval_index_paths
    with open(data_path / 'trainval_index_names.pkl', 'rb') as f:
        trainval_index_paths = pickle.load(f)

    global trainval_hash_to_path
    trainval_hash_to_path = {hashlib.md5(str(image_path).encode("utf-8")).hexdigest(): image_path for image_path in
                             trainval_index_paths}

    global trainval_index_hashes
    trainval_index_hashes = list(trainval_hash_to_path.keys())

    # Test ---------------------------
    global test_index_features
    test_index_features = torch.load(
        data_path / 'test_index_features.pt', map_location=device).type(data_type).cpu()

    global test_index_paths
    with open(data_path / 'test_index_names.pkl', 'rb') as f:
        test_index_paths = pickle.load(f)

    global test_hash_to_path
    test_hash_to_path = {hashlib.md5(str(image_path).encode("utf-8")).hexdigest(): image_path for image_path in
                         test_index_paths}

    global test_index_hashes
    test_index_hashes = list(test_hash_to_path.keys())

    # Combined
    global global_index_features
    global_index_features = torch.vstack((trainval_index_features, test_index_features)).type(data_type).to(device)

    global global_index_paths
    global_index_paths = trainval_index_paths + test_index_paths

    global global_hash_to_path
    global_hash_to_path = {}
    global_hash_to_path.update(trainval_hash_to_path)
    global_hash_to_path.update(test_hash_to_path)

    global global_index_hashes
    global_index_hashes = trainval_index_hashes + test_index_hashes

    # Load CLIP model and Combiner networks
    global clip_model
    global clip_preprocess
    clip_model, clip_preprocess = clip.load("RN50x4")
    clip_model = clip_model.eval().to(device)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=PORT)
