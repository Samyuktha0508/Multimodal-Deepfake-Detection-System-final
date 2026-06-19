import os
import cv2
import torch
import zipfile
import librosa
import numpy as np
import tensorflow_addons
import tensorflow as tf
from facenet_pytorch import MTCNN
from rawnet import RawNet



#Set random seed for reproducibility.
tf.random.set_seed(42)

local_zip = "./efficientnet-b0.zip"
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall()
zip_ref.close()


# Load models.
model = tf.keras.models.load_model("efficientnet-b0/")



class DetectionPipeline:
    """Pipeline class for detecting faces in the frames of a video file."""

    def __init__(self, n_frames=None, batch_size=60, resize=None, input_modality = 'video'):
        """Constructor for DetectionPipeline class.

        Keyword Arguments:
            n_frames {int} -- Total number of frames to load. These will be evenly spaced
                throughout the video. If not specified (i.e., None), all frames will be loaded.
                (default: {None})
            batch_size {int} -- Batch size to use with MTCNN face detector. (default: {32})
            resize {float} -- Fraction by which to resize frames from original prior to face
                detection. A value less than 1 results in downsampling and a value greater than
                1 result in upsampling. (default: {None})
        """
        self.n_frames = n_frames
        self.batch_size = batch_size
        self.resize = resize
        self.input_modality = input_modality

    def __call__(self, filename):
        """Load frames from an MP4 video and detect faces.

        Arguments:
            filename {str} -- Path to video.
        """
        # Create video reader and find length
        if self.input_modality == 'video':
            print('Input modality is video.')
            v_cap = cv2.VideoCapture(filename)
            v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Pick 'n_frames' evenly spaced frames to sample
            if self.n_frames is None:
                sample = np.arange(0, v_len)
            else:
                sample = np.linspace(0, v_len - 1, self.n_frames).astype(int)

            # Loop through frames
            faces = []
            frames = []
            for j in range(v_len):
                success = v_cap.grab()
                if j in sample:
                    # Load frame
                    success, frame = v_cap.retrieve()
                    if not success:
                        continue
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Resize frame to desired size
                    if self.resize is not None:
                        frame = frame.resize([int(d * self.resize) for d in frame.size])
                    frames.append(frame)

                    # When batch is full, detect faces and reset frame list
                    if len(frames) % self.batch_size == 0 or j == sample[-1]:
                        face2 = cv2.resize(frame, (224, 224))
                        faces.append(face2)

            v_cap.release()
            return faces

        elif self.input_modality == 'image':
            print('Input modality is image.')
            #Perform inference for image modality.
            print('Reading image')
            # print(f"Image path is: {filename}")
            image = cv2.cvtColor(filename, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))

            # if not face.any():
            #     print("No faces found...")

            return image
        
        elif self.input_modality == 'audio':
            print("INput modality is audio.")

            #Load audio.
            x, sr = librosa.load(filename)
            x_pt = torch.Tensor(x)
            x_pt = torch.unsqueeze(x_pt, dim = 0)
            return x_pt
        
        else:
            raise ValueError("Invalid input modality. Must be either 'video' or image")

detection_video_pipeline = DetectionPipeline(n_frames=5, batch_size=1, input_modality='video')
detection_image_pipeline = DetectionPipeline(batch_size = 1, input_modality = 'image')

def deepfakes_video_predict(input_video):

    faces = detection_video_pipeline(input_video)
    total = 0
    real_res = []
    fake_res = []

    for face in faces:

        face2 = face/255
        pred = model.predict(np.expand_dims(face2, axis=0))[0]
        real, fake = pred[0], pred[1]
        real_res.append(real)
        fake_res.append(fake)

        total+=1

        pred2 = pred[1]

        if pred2 > 0.5:
          fake+=1
        else:
          real+=1
    real_mean = np.mean(real_res)
    fake_mean = np.mean(fake_res)
    print(f"Real Faces: {real_mean}")
    print(f"Fake Faces: {fake_mean}")
    text = ""

    if real_mean >= 0.5:
        text = "The video is REAL. \n Deepfakes Confidence: " + str(round(100 - (real_mean*100), 3)) + "%"
    else:
        text = "The video is FAKE. \n Deepfakes Confidence: " + str(round(fake_mean*100, 3)) + "%"

    return text


def deepfakes_image_predict(input_image):
    faces = detection_image_pipeline(input_image)
    face2 = faces/255
    pred = model.predict(np.expand_dims(face2, axis = 0))[0]
    real, fake = pred[0], pred[1]
    if real > 0.5:
        text2 = "The image is REAL. \n Deepfakes Confidence: " + str(round(100 - (real*100), 3)) + "%"
    else:
        text2 = "The image is FAKE. \n Deepfakes Confidence: " + str(round(fake*100, 3)) + "%"
    return text2

def load_audio_model():
    d_args = {
  "nb_samp": 64600,
  "first_conv": 1024,
  "in_channels": 1,
  "filts": [20, [20, 20], [20, 128], [128, 128]],
  "blocks": [2, 4],
  "nb_fc_node": 1024,
  "gru_node": 1024,
  "nb_gru_layer": 3,
  "nb_classes": 2}
    
    model = RawNet(d_args = d_args, device='cpu')

    #Load ckpt.
    model_dict = model.state_dict()
    ckpt = torch.load('RawNet2.pth', map_location=torch.device('cpu'))
    model.load_state_dict(ckpt, model_dict)
    return model

audio_label_map = {
    0: "Real audio",
    1: "Fake audio"
}

def deepfakes_audio_predict(input_audio):
    #Perform inference on audio.
    x, sr = input_audio
    x_pt = torch.Tensor(x)
    x_pt = torch.unsqueeze(x_pt, dim = 0)

    #Load model.
    model = load_audio_model()

    #Perform inference.
    grads = model(x_pt)

    #Get the argmax.
    grads_np = grads.detach().numpy()
    result = np.argmax(grads_np)

    return audio_label_map[result]
