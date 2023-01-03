import argparse
import cv2
import insightface
from insightface.app import FaceAnalysis


assert insightface.__version__>='0.3'

parser = argparse.ArgumentParser(description='insightface app test')
# general
parser.add_argument('--ctx', default=0, type=int, help='ctx id, <0 means using cpu')
parser.add_argument('--det-size', default=640, type=int, help='detection size')
args = parser.parse_args()


def detect_face(img):
    DEFAULT_MP_NAME = 'buffalo_l'
    DEFAULT_ROUTE = '~/.insightface'
    app = FaceAnalysis(name=DEFAULT_MP_NAME, route=DEFAULT_ROUTE, allowed_modules='detection', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=args.ctx, det_size=(args.det_size, args.det_size))

    faces = app.get(img)

    return faces



