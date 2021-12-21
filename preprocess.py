"""
This class can;
    - load photos from database
    - detect face and crop with opencv
    - create embedding vector from face.
"""
import os
import cv2
import numpy as np
from keras.models import load_model
from mtcnn.mtcnn import MTCNN
import config
import timeit

class Preprocess:
    
    def __init__(self, database_path, created_database):
        
        self.path = database_path
        self.path2 = created_database
        
        self.model =  load_model('model/facenet_keras.h5')
        
        self.face_cascade = cv2.CascadeClassifier("model/haarcascade_frontalface_default.xml")

        print("[Log] Preprocess object was created.")

    def load_images(self, is_db):
        
        if is_db :
            #read embedded arrays
            database = {}
            folders = os.listdir(self.path2)
            for folder in folders:
                database[folder] = []
                files = os.listdir(os.path.join(self.path2, folder))
                for file in files:
                    filepath = os.path.join(self.path2,folder,file)
                    a_file = open(filepath,"r")
                    database[folder].append(np.loadtxt(a_file, dtype=float))
                    a_file.close()
            print("[Log] Database was loaded.")
                    
            
        else:
            #create database
            database = {}
            folders = os.listdir(self.path)
            for folder in folders:
                database[folder] = []
                files = os.listdir(os.path.join(self.path,folder))
                for file in files:
                    filepath =  os.path.join(self.path,folder,file)
                    img = cv2.imread(filepath)
                    if config.detectionAlgorithm == "mtcnn":
                        (faces, _) = self.getFaceMTCNN(img)
                    else:
                        (faces, _) = self.getFaceOpencv(img)
                    if faces != None:
                        for face in faces:
                            database[folder].append(self.embedding(face))
            print("[Log] Database was created.")
            
            #Store the embedded values
            for key in database.keys():
                for i, val in enumerate(database[key]):
                    a_file = open(self.path2+"/"+key+"/"+str(i+1)+".txt", "w")
                    np.savetxt(a_file, val)
                    a_file.close()
            print("[Log] Database was stored.")
        
            
        return database
        

    def embedding(self,img):
        """ embed face with facenet model """
        img = img[...,::-1]
        img = np.around(np.transpose(img, (0,1,2))/255.0, decimals=12)
        img = np.array([img])
        
        #not much time wasting
        embedding = self.model.predict_on_batch(img)
        
        return embedding[0]
    
    
    i = 1
    def getFaceOpencv(self, img):
        
        face_list = []
        face_coor = []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        #face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = self.face_cascade.detectMultiScale(gray, 1.05, 18)
        #bboxes = classifier.detectMultiScale(pixels, 1.05, 8)
        if len(faces)!=0:
            for (x, y, w, h) in faces:
                x1 = x
                y1 = y
                x2 = x+w
                y2 = y+h
                face_image = img[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]  
                face_image = cv2.resize(face_image, (160, 160))  
                face_list.append(face_image)
                face_coor.append((x1,y1,x2,y2))
                #print("face:")
                #print("face_list", face_list[0][0], "face_coor", face_coor)
                #cv2.imwrite("yuzler/"+str(Preprocess.i)+'.jpg', face_image)
                Preprocess.i +=1
                
                
            return (face_list,face_coor)
        else:
            return (None,None)
        
    def getFaceMTCNN(self, img):
        face_list = []
        face_coor = []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        detector = MTCNN()
        #time wasting
        faces = detector.detect_faces(img)
        
        if len(faces)!=0:
            for face in faces:
                x, y, w, h = face['box']
                x2 = x+w
                y2 = y+h
                face_image = img[max(0, y):min(height, y2), max(0,x):min(width,x2)]
                face_image = cv2.resize(face_image, (160,160))
                face_list.append(face_image)
                face_coor.append((x,y,x2,y2))
                
                #cv2.imwrite("yuzler/"+str(Preprocess.i)+'.jpg', face_image)
                Preprocess.i +=1
            return (face_list, face_coor)
        else:
            return (None,None)
        
    def euclid_distance(self, input_embed, db_embed):
        """ calculate euclidan distance between two embeded vector """
        return np.linalg.norm(db_embed-input_embed)

    