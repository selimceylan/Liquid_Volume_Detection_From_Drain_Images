import os
from flask import Flask, request, send_file
from InferenceCode import Inference
from keras import backend as K
import skimage.io

UPLOAD_FOLDER = './upload'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file1' not in request.files:
            return 'there is no file1 in form!'
        file1 = request.files['file1']
        path = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
        file1.save(path)
        CUSTOM_MODEL_PATH = "C:\\Users\\slmcy\\Desktop\\Graduation_Project\\First_Train_Results\\logs2\\drain20211219T1936\\mask_rcnn_drain_0012.h5"
        MODEL_DIR = "C:\\Users\\slmcy\\Desktop\\Graduation_Project\\First_Train_Results\\logs2\\drain20211219T1936"
        IMAGE_DIR = path
        K.clear_session()
        img = Inference(CUSTOM_MODEL_PATH, MODEL_DIR, IMAGE_DIR)
        K.clear_session()
        os.remove(path)
        skimage.io.imsave("./upload/result.jpg", img)
        return send_file("./upload/result.jpg", mimetype='image/gif')
        # return path

        return 'ok'
    return '''
    <h1>Upload new File</h1>
    <form method="post" enctype="multipart/form-data">
      <input type="file" name="file1">
      <input type="submit">
    </form>
    '''


# @app.route("/image")
# def processimage():
#     CUSTOM_MODEL_PATH = "C:\\Users\\slmcy\\Desktop\\Graduation_Project\\First_Train_Results\\logs2\\drain20211219T1936\\mask_rcnn_drain_0012.h5"
#     MODEL_DIR = "C:\\Users\\slmcy\\Desktop\\Graduation_Project\\Train_Results_drain2\\drain20220115T1937"
#     IMAGE_DIR = "./upload/photo5949750180502943546.jpg"
#     img = Inference(CUSTOM_MODEL_PATH,MODEL_DIR,IMAGE_DIR)
#     cv2.imwrite("./upload/result.jpg",img)
#     return send_file("./upload/result.jpg", mimetype='image/gif')


if __name__ == '__main__':
    app.run()
