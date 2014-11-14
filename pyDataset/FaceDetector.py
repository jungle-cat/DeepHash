'''
Created on Nov 14, 2014

@author: Feng
'''

import cv2

class Face(object):
    def __init__(self, image, rect=None):
        self.image = image
        if rect is not None:
            self.faces = [self.image[r[0]:r[1]+r[0], r[2]:r[3]+r[2]] for r in rect]
        else:
            self.faces = [self.image]
        self.faces = [cv2.resize(face, (120,120), interpolation=cv2.INTER_CUBIC) for face in self.faces]
    

class FaceDetector(object):
    def __init__(self, face_casfile, eye_casfile=None):
        self.face_cascade = cv2.CascadeClassifier(face_casfile)
        if eye_casfile is not None:
            self.eye_cascade = cv2.CascadeClassifier(eye_casfile)
        else:
            self.eye_cascade = None
    
    def detect(self, image):
        if isinstance(image, str):
            image = cv2.imread(image)
        
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        else:
            gray = image
        
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        sorted_faces = sorted(faces,key=lambda face: face[2]*face[3])
        
        return Face(image, sorted_faces)
        
            
            