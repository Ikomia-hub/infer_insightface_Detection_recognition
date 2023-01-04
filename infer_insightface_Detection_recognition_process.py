# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
#
# This file is part of the IkomiaStudio software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from ikomia import core, dataprocess
import copy
import cv2
import os
import argparse
import insightface
import numpy as np
from insightface.app import FaceAnalysis
from infer_insightface_Detection_recognition.detection.scrfd_detection_model import SCRFD
from infer_insightface_Detection_recognition.detection.face_detection import detect_face
# Your imports below


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferInsightfaceDetectionRecognitionParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.taskName = "detection"
        self.modelName = ""
        self.modelFile = ""

    def setParamMap(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.taskName = param_map["taskName"]
        self.modelName = param_map["modelName"]
        self.modelFile = param_map["modelFile"]
        pass

    def getParamMap(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = core.ParamMap()
        param_map["taskName"] = self.taskName
        param_map["modelName"] = self.modelName
        param_map["modelFile"] = self.modelFile
        return param_map


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferInsightfaceDetectionRecognition(dataprocess.C2dImageTask):

    def __init__(self, name, param):
        dataprocess.C2dImageTask.__init__(self, name)
        # Add input/output of the process here
        self.addInput(dataprocess.CImageIO())
        # Add object detection output
        self.addOutput(dataprocess.CObjectDetectionIO())
        self.obj_detect_output = None
        # Add graphics output
        self.addOutput(dataprocess.CGraphicsOutput())

        # Create parameters class
        if param is None:
            self.setParam(InferInsightfaceDetectionRecognitionParam())
        else:
            self.setParam(copy.deepcopy(param))

    def getProgressSteps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def draw_on(self, id, box, score):
        w = abs(box[2] - box[0])
        h = abs(box[3] - box[1])
        self.output_obj_detect.addObject(id, "face", float(score), boxX=float(box[0]), boxY=float(box[1]),
                                    boxWidth=float(w), boxHeight=float(h), color=[250, 0, 0])

    def run(self):
        # Core function of your process
        # Call beginTaskRun for initialization
        self.beginTaskRun()

        # Get input :
        input = self.getInput(0)
        input2 = self.getInput(2)

        # Get parameters :
        param = self.getParam()

        # face detection task
        if param.taskName == 'face detection':
            # Get output :
            self.output_obj_detect = self.getOutput(1)
            self.output_obj_detect.init("face_detection", 0)
            # Forward input image
            self.forwardInputImage(0, 0)
            # Get image from input/output (numpy array):
            srcImage = input.getImage()
            # using scrfd model
            if param.modelName == 'scrfd face detection' and os.path.isfile(param.modelFile):
                detector = SCRFD(model_file=param.modelFile)
                detector.prepare(-1)
                bboxes, kpss = detector.detect(srcImage, thresh=0.5, input_size=(640, 640))
                if kpss is not None:
                    for i in range(bboxes.shape[0]):
                        bbox = bboxes[i]
                        score = bbox[4]
                        self.draw_on(i + 1, bbox, float(score))

            # using retinaface model
            elif param.modelName == 'retinaface detection':
                pass
            else:
                # using default face detection : buffalo_l
                print('default face detection')
                faces = detect_face(srcImage)
                for i in range(len(faces)):
                    face = faces[i]
                    box = face.bbox.astype(int)
                    score = face.det_score
                    self.draw_on(i + 1, box, score)
                    # add landmarks
                    if face.kps is not None:
                        kps = face.kps.astype(int)
                        for l in range(kps.shape[0]):
                            color = (0, 0, 255)
                            if l == 0 or l == 3:
                                color = (0, 255, 0)
                            cv2.circle(srcImage, (kps[l][0], kps[l][1]), 1, color,
                                       2)

        # person detection task
        elif param.taskName == 'person detection':

            # Get output :
            self.output_obj_detect = self.getOutput(1)
            self.output_obj_detect.init("face_detection", 0)
            # Forward input image
            self.forwardInputImage(0, 0)

            # Get image from input/output (numpy array):
            srcImage = input.getImage()
            # get model
            if os.path.isfile(param.modelFile):
                detection_model = param.modelFile
            else:
                detection_model = 'scrfd_person_2.5g.onnx'
            # detect person
            detector = insightface.model_zoo.get_model(detection_model, download=True)
            detector.prepare(0, nms_thresh=0.5, input_size=(640, 640))
            bboxes, kpss = detector.detect(srcImage)
            for i in range(bboxes.shape[0]):
                bbox = bboxes[i]
                score = bbox[4]
                self.draw_on(i + 1, bbox, score)


        elif param.taskName == 'recognition':
            # Get output :
            output = self.getOutput(0)
            # Get image from input/output (numpy array):
            img1 = input.getImage()
            img2 = input2.getImage()
            # resize image
            img1 = cv2.resize(img1, (112, 112))
            img2 = cv2.resize(img2, (112, 112))
            # detect face on each image
            app = FaceAnalysis(providers=['CPUExecutionProvider'])
            app.prepare(ctx_id=0, det_size=(640, 640))
            faces1 = app.get(img1)
            faces2 = app.get(img2)
            # Recognition Model
            if os.path.isfile(param.modelFile):
                recognition_model = param.modelFile
            else:
                recognition_model = 'buffalo_l/w600k_r50.onnx'

            handler = insightface.model_zoo.get_model(recognition_model, download=True)
            handler.prepare(ctx_id=0)
            img_feat = handler.get(img1, faces1[0])
            img_feat2 = handler.get(img2, faces2[0])
            simularity = handler.compute_sim(img_feat, img_feat2)
            # Create a blank 300x300 black image
            image = np.zeros((500, 500, 3), np.uint8)
            if simularity > 0.3:
                # Fill image with red color(set each pixel to green)
                image[:] = (0, 255, 0)
                # Using cv2.putText() method
                text = f'The same person, \n score:{float(simularity)}'
                y0, dy = 150, 100

                for i, line in enumerate(text.split('\n')):
                    y = y0 + i * dy
                    image = cv2.putText(image, line, (0, y),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                # Fill graphics output
                #graphics_output = self.getOutput(2)
                #graphics = graphics_output.addText(str(text), 0, 0)
                #dataprocess.CImageIO.getImageWithGraphics(graphics)
            else:
                # Fill image with red color(set each pixel to red)
                image[:] = (255, 0, 0)
                text = f'Not the same person, \n score:{float(simularity)}'

                y0, dy = 150, 100
                for i, line in enumerate(text.split('\n')):
                    y = y0 + i * dy
                    image = cv2.putText(image, line, (0, y),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                #graphics_output = self.getOutput(2)
                #graphics_output.addText(str(text), 0, 0)

            # Set image of input/output (numpy array):
            output.setImage(image)

        elif param.taskName == 'alignment':
            # Get output :
            output = self.getOutput(0)
            app = FaceAnalysis(allowed_modules=['detection', 'landmark_2d_106'])
            app.prepare(ctx_id=0, det_size=(640, 640))
            img = input.getImage()
            faces = app.get(img)
            # assert len(faces)==6
            tim = img.copy()
            color = (250, 0, 250)
            for face in faces:
                lmk = face.landmark_2d_106
                lmk = np.round(lmk).astype(int)
                for i in range(lmk.shape[0]):
                    p = tuple(lmk[i])
                    cv2.circle(tim, p, 2, color, 1, cv2.LINE_AA)

            output.setImage(tim)

        # Step progress bar:
        self.emitStepProgress()

        # Call endTaskRun to finalize process
        self.endTaskRun()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferInsightfaceDetectionRecognitionFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_insightface_Detection_recognition"
        self.info.shortDescription = "face detection, recognition, alignment models"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python"
        self.info.version = "1.0.0"
        # self.info.iconPath = "your path to a specific icon"
        self.info.authors = "Jia Guo, Jiankang Deng, Xiang An, Jack Yu, Baris Gecer"
        self.info.article = "Sample and Computation Redistribution for Efficient Face Detection"
        self.info.year = 2021
        self.info.license = "MIT License"
        # Keywords used for search
        self.info.keywords = "face detection, person detection,recognition, alignment"

    def create(self, param=None):
        # Create process object
        return InferInsightfaceDetectionRecognition(self.info.name, param)
