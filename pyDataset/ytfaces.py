'''
Created on Nov 14, 2014

@author: Feng
'''

try:
    from PIL import Image
except ImportError:
    import Image

import os, random, numpy

from pyDataset.FaceDetector import FaceDetector

def build_dataset(base_dir, num_people, num_pfaces, num_max):
    
    #TODO specify cascades xml file path
    detector = FaceDetector('data/haarcascade_frontalface_default.xml')
    
    people_names = os.listdir(base_dir)
    random.shuffle(people_names)
    
    # list all posible person names, maximumly n people
    n = len(people_names)
    if num_people > 0 and n > num_people:
        n = num_people
        people_names = people_names[0:n]
    
    face_images = []
    face_ids = []
    
    for i, name in enumerate(people_names):
        desc_path = os.path.join(base_dir, name)
        print 'load path %s' % desc_path
        
        # list all face image of person name
        image_names = []
        for item in os.listdir(desc_path):
            image_list = [os.path.join(item, im) for im in os.listdir(os.path.join(desc_path, item))]
            image_names.extend(image_list)
        random.shuffle(image_names)
        
        n_images = len(image_names)
        if n_images > num_pfaces:
            n_images = num_pfaces
            image_names = image_names[0:n_images]
        
        
        # detect faces of image list
        for image_name in image_names:
            image_path = os.path.join(desc_path, image_name)
            face_rets = detector.detect(image_path)
            if face_rets is not None and len(face_rets.faces) > 0:
                face_images.append(face_rets.faces[0])
                face_ids.append(i+1)
    
    assert len(face_images) == len(face_ids)
    
    # only num_max faces are used
    # TODO random select num_max face to return
    ntotal_faces = len(face_images)
    if ntotal_faces > num_max and num_max > 0:
        ntotal_faces = num_max
        face_images = face_images[0:ntotal_faces]
        face_ids = face_ids[0:ntotal_faces]
        
    # convert list of faces to tensor like numpy.array
    faces = numpy.asarray(face_images)
    ids = numpy.asarray(face_ids)
    
    return (faces, ids)
        

if __name__ == '__main__':
    base_path = r'/mnt/UDisk/Database/YouTube\ Faces/aligned_images_DB/'
    rets = build_dataset(base_path, num_people=1000, num_pfaces=200, num_max=100000)
