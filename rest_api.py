import os
from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from video_model import VideoModel
from image_model import ImageModel

app = Flask(__name__)
api = Api(app)

class ImageCaptioning(Resource):
    def post(self):
        if "file" not in request.files:
            return "No file found"

        user_file = request.files["file"]
        temp = request.files["file"]
        if user_file.filename == "":
            return "file name not found"
        else:
            path = os.getcwd()+'\\static_files\\images\\'+user_file.filename
            user_file.save(path)
            caption = self.get_caption(path)
            return jsonify({
                "status": "success",
                "caption": caption,
                })

    def get_caption(self, path):
        image_model = ImageModel(path)
        caption = image_model.get_pred()

        return caption

class VideoCaptioning(Resource):
    def post(self):
        if "file" not in request.files:
            return "No file found"

        user_file = request.files["file"]
        temp = request.files["file"]
        if user_file.filename == "":
            return "file name not found"
        else:
            path = os.getcwd()+'\\static_files\\videos\\'+user_file.filename
            user_file.save(path)
            print(path)
            caption = self.get_caption(path)
            return jsonify({
                "status": "success",
                "caption": caption,
                })

    def get_caption(self, path):
        video_model = VideoModel(path)
        caption = video_model.get_pred()

        return caption

class ImagePrediction(Resource):
    def get(self, image_name):
        path = "C:\\Users\\Pulkit Gupta\\Documents\\video-captioning\dataset\\flickr30k_images\\flickr30k_images\\"+image_name+'.jpg'
        caption = self.get_caption(path)
        return {'result': caption}

    def get_caption(self, image_name):
        image_model = ImageModel(image_name)
        caption = image_model.get_pred()

        return caption

class VideoPrediction(Resource):
    def get(self, video_name):
        path = "C:\\Users\\Pulkit Gupta\\Documents\\video-captioning\dataset\\msvd_videos\\msvd_videos\\"+video_name+'.avi'
        caption = self.get_caption(path)
        return {'result': caption}

    def get_caption(self, path):
        video_model = VideoModel(path)
        caption = video_model.get_pred()

        return caption

api.add_resource(ImageCaptioning, '/image')
api.add_resource(VideoCaptioning, '/video')
api.add_resource(ImagePrediction, '/imagepred/<string:image_name>')
api.add_resource(VideoPrediction, '/videopred/<string:video_name>')

if __name__=='__main__':
    app.run(debug=True, host='192.168.0.106', port=5000)
