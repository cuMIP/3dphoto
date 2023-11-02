import os
DATA_DIR = './data/'
MODEL_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), './NormativePCAModel/')
CLASSIFIER_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), './CraniosynostosisClassifier/')

CRANIOFACIAL_LANDMARKING_MODEL_PATH = os.path.join(DATA_DIR, 'CraniofacialLandmarkingModel.dat')
CRANIOFACIAL_LANDMARKING_NOTEXTURE_MODEL_PATH = os.path.join(DATA_DIR, 'CraniofacialLandmarkingModel-notexture.dat')
GLABELLA_CRANIALBASE_LANDMARKS_PATH = os.path.join(DATA_DIR, 'landmarks_ct_registration.vtp')
EURYON_CRANIALBASE_LANDMARKS_PATH = os.path.join(DATA_DIR, 'landmarks_3dphoto_registration.vtp')
CUTTING_LANDMARKS_PATH = os.path.join(DATA_DIR,'landmarks_for_cutting.vtp')

TEMPLATE_VOLUME = 898.32