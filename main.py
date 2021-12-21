from  preprocess import Preprocess
from keras.models import load_model
import cv2
import config
import timeit
import time
from FPS import FPS
from webcamvideoStream import WebcamVideoStream
import argparse
import datetime

class FaceRecognition:
    def __init__(self):
        
        ap = argparse.ArgumentParser()
        ap.add_argument("-n", "--num-frames", type=int, default=100,
        	help="# of frames to loop over for FPS test")
        ap.add_argument("-d", "--display", type=int, default=-1,
        	help="Whether or not frames should be displayed")
        args = vars(ap.parse_args())
        
        
        """initialize dataset and load model"""
        self.model = load_model(config.model_path)
        print("[Log] Pretrained model was loaded.")
        
        self.preprocess = Preprocess(database_path=config.database_path, created_database=config.database_path2)
        print("[Log] Preprocess object was created.")
        
        self.database = self.init_database()
        
        
    def init_database(self):
        """ initilize face database"""
        if config.is_db_created:
            return self.preprocess.load_images(config.is_db_created)
        else:
            return self.preprocess.load_images(False)
        print("[Log] Database initialized.",flush=True)

    
    def recognize_faces_in_video(self, videopath):
        
        vc = cv2.VideoCapture(videopath)
        
        while vc.isOpened():
            _, frame = vc.read()
            
            height, width, channels = frame.shape
            #print(frame.shape) > (400, 720, 3)
            
            faces, coordinates = self.preprocess.getFace(frame)    
            if faces == None:
                continue
            try:
                # Loop through all the faces
                for i, face in enumerate(faces):
                     
                    embeded_face = self.preprocess.embedding(face)
                    name, similarity =  self.findFaceInDB(embeded_face)
                    #print(name,"-",similarity)
                    if similarity <= config.threshold:
                        x1, y1, x2, y2 = coordinates[i]
                        #print(coordinates) > [(152, 72, 297, 217)]
                        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
                        cv2.putText(frame,name, (x1-10, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
            except TypeError:
              None          
                        
                        
            key = cv2.waitKey(100)
            cv2.imshow("Face Recognizer", frame)
            if key == 27: # exit on ESC
                break
        
        vc.release()
        cv2.destroyAllWindows()
        
    def recognize_faces_in_video2(self, videopath):
        vc = cv2.VideoCapture(0)
  
        while(True):
              
            # Capture the video frame
            # by frame
            ret, frame = vc.read()      
            
          
            # Display the resulting frame
            #cv2.imshow('frame', frame)
            
            
            height, width, channels = frame.shape
            
            faces, coordinates = self.preprocess.getFace(frame) 
            
            if faces == None:
                continue
            
            for i, face in enumerate(faces):
                
                         
                embeded_face = self.preprocess.embedding(face)
                name, similarity =  self.findFaceInDB(embeded_face)
                #print(name,"-",similarity)
                if similarity <= config.threshold:
                    print("similar")
                    x1, y1, x2, y2 = coordinates[i]
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
                    cv2.putText(frame,name, (x1-10, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
            
            
            #cv2.imshow('frame', frame)
            
            # the 'q' button is set as the
            # quitting button you may use any
            # desired button of your choice
            #if cv2.waitKey(1) & 0xFF == ord('q'):
                #,break
          
        # After the loop release the cap object
        vc.release()
        # Destroy all the windows
        cv2.destroyAllWindows()
        
        
    def recognize_faces_in_photo(self, photo_path):
        frame = cv2.imread(photo_path,1)
        faces, coordinates = self.preprocess.getFace(frame) 
        
        #if faces == None:
            #continue
        try:    
            for i, face in enumerate(faces):        
                         
                embeded_face = self.preprocess.embedding(face)
                name, similarity =  self.findFaceInDB(embeded_face)
                print(name,"-",similarity)
                if similarity <= config.threshold:
                    print("benzer")
                    x1, y1, x2, y2 = coordinates[i]
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
                    cv2.putText(frame,name, (x1-10, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
        except TypeError:
            print("TypeError: Check list of indices Yani yüz yok")
                    
    
    
    def recognize_faces_in_camera(self):
        vc = cv2.VideoCapture(0)
        start = datetime.datetime.now()
        
        numFrames = 1
        
        while(True):
            #time.sleep(0.005) #works but can't find face fast
            numFrames = numFrames + 1
            
            _, frame = vc.read()    
            
            #print(frame.shape) > (480, 640, 3)
            #if(numFrames>0):
            if(config.detectionAlgorithm == 'open-cv'):
                faces, coordinates = self.preprocess.getFaceOpencv(frame) 
            else:                
                faces, coordinates = self.preprocess.getFaceMTCNN(frame)               
            
                      
            try:                
                for i, face in enumerate(faces):
                    
                    embeded_face = self.preprocess.embedding(face)
                    name, similarity =  self.findFaceInDB(embeded_face)  
                    
                    if similarity <= config.threshold:
                        x1, y1, x2, y2 = coordinates[i]
                        #print(coordinates) > [(179, 132, 430, 383)]
                        #print(name, similarity)
                        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
                        cv2.putText(frame,name, (x1-10, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
                        
            except TypeError:
                None
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break  
            cv2.imshow('frame', frame)
            
        end = datetime.datetime.now()  
        print(numFrames)
        elapsed = (end-start).total_seconds()
        print(elapsed)
        print(numFrames/elapsed)
        vc.release()
        cv2.destroyAllWindows()
          
    def recognize_faces_in_cameraThread(self):
        vc = cv2.VideoCapture(0)
        #vs = WebcamVideoStream(src=0).start()
        fps = FPS().start()
        
        islem = 0 
        while(True):
            islem = islem+1
            time.sleep(0.05) #works but can't find face fast
            _, frame = vc.read()   
            #frame = vs.read()
            
            
            #print(frame.shape) > (480, 640, 3)
            
            if(config.detectionAlgorithm == 'open-cv'):
                faces, coordinates = self.preprocess.getFaceOpencv(frame) 
            else:                
                faces, coordinates = self.preprocess.getFaceMTCNN(frame)               
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break            
            try:                
                for i, face in enumerate(faces):
                    time.sleep(1)
                    embeded_face = self.preprocess.embedding(face)
                    name, similarity =  self.findFaceInDB(embeded_face)  
                    
                    if similarity <= config.threshold:
                        x1, y1, x2, y2 = coordinates[i]
                        #print(coordinates) > [(179, 132, 430, 383)]
                        print(name, similarity)
                        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
                        cv2.putText(frame,name, (x1-10, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
                        
            except TypeError:
                None
            cv2.imshow('frame', frame)
            
            fps.update()
            
        fps.stop()
        print("islem ", islem)
        print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

            
        
    def findFaceInDB(self, embedded_face):
        """ This method finds person in person database."""    
        who, similarity = None, None
        min_similarity = 20
        #we shouldn't use for loops here
        #önce tek kişi ile hız denemesi yap
        persons = list(self.database.keys())
        for person in persons:
            for vec in self.database[person]:
                sim = self.preprocess.euclid_distance(embedded_face,vec)
                if sim < min_similarity:
                    min_similarity = sim
                    who = person
                    
            
        return who, min_similarity


                
#bboxes = classifier.detectMultiScale(pixels, 1.05, 8)

if __name__ == "__main__":
    #create_database    
    FR = FaceRecognition()
    #FR.recognize_faces_in_photo(config.photo_path)
    #FR.recognize_faces_in_video(config.video_path)
    FR.recognize_faces_in_camera()



