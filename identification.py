import numpy as np
import face_recognition as fr
import cv2
import os 

os.system('cls')
pics = os.listdir('./known_faces')
known_faces_encodings = []
known_faces_names = []


pics_file_path = input("Enter the complete path of the pics file : ")


video_capture = cv2.VideoCapture(0)



while True: 
    os.system('cls')
    pics = os.listdir('./known_faces')
    known_faces_encodings = []
    known_faces_names = []
    for pic in pics:
        image = fr.load_image_file("./known_faces/"+str(pic))
        image_encoding = fr.face_encodings(image)[0]
        known_faces_encodings.append(image_encoding)
        if ".jpg" in pic:
            name = pic.replace(".jpg", "")
            known_faces_names.append(name)
        elif ".png" in pic:
            name = pic.replace(".png","")
            known_faces_names.append(name)



    print("\n"+
    "================Menu================\n"+
    "| a. List known people             |\n"+
    "| b. Add a person                  |\n"+
    "| c. Delete a person               |\n"+
    "| d. Launch the face recognition   |\n"+
    "| e. Quit the program              |\n"+
    "====================================\n"
    )


    answer = input("Enter your choice : ")

    #list every persons in the dataset
    if(answer == 'a'):
        known_faces_encodings = []
        known_faces_names = []  
        for pic in pics:
            image = fr.load_image_file("./known_faces/"+str(pic))
            image_encoding = fr.face_encodings(image)[0]
            known_faces_encodings.append(image_encoding)
            if ".jpg" in pic:
                name = pic.replace(".jpg", "")
                known_faces_names.append(name)
            elif ".png" in pic:
                name = pic.replace(".png","")
                known_faces_names.append(name)

        print("The people known by the algorithm are :")
        for people in known_faces_names:
            print(people)
        input("Press enter to continue...")


    #Add a person to the dataset with the webcam
    if (answer == 'b'):
        print("The people known by the algorithm are :")
        for people in known_faces_names:
            print(people)

        answer = input("Are you in the list? (Y/N) :")
        if answer=="N" or answer=="n":
            answer=input("Do you want to register yourself? (Y/N) :")
            if answer=="Y" or answer=="y":
                name=input("Enter your name : ")
                while(name in known_faces_names):
                    name=input("This name already exist(add your family name) : ")

                print("Press 'space' to take the the picture")


                while True:
                    ret, photo = video_capture.read()
                    cv2.imshow('photo',photo)

                    k = cv2.waitKey(1)
                    if k%256 == 32:
                        img_name = name+".jpg"
                        cv2.imwrite(os.path.join(pics_file_path , img_name),photo)
                        break

                cv2.destroyAllWindows()
            input("Press enter to continue...")


    #delete a known person from the Dataset
    if (answer == 'c'):
        print("The people known by the algorithm are :")
        for people in known_faces_names:
            print(people)

        suppr_name = input("Who would you like to delete? : ")
        if (os.path.isfile('./known_faces/'+ suppr_name+'.png')):
            os.remove("{}\{}.png".format(pics_file_path,suppr_name))
            print("File deleted")
        elif (os.path.isfile('./known_faces/'+ suppr_name+'.jpg')):
            os.remove("{}\{}.jpg".format(pics_file_path,suppr_name))
            print("File deleted")
        input("Press enter to continue...")


    if (answer == 'd'):

        while True:
            ret, frame = video_capture.read()
            rgb_frame = frame[:,:,::-1]
            cv2.imshow('Cam',frame)
            face_locations = fr.face_locations(rgb_frame)
            face_encodings = fr.face_encodings(rgb_frame,face_locations)
            cv2.imshow('Cam',frame)
            for (top,right,bottom,left), face_encoding in zip(face_locations,face_encodings):
                matches = fr.compare_faces(known_faces_encodings,face_encoding)
                name = "Unknown"


            cv2.imshow('Cam',frame)

            face_distance = fr.face_distance(known_faces_encodings, face_encoding)

            cv2.imshow('Cam',frame)
            
            #un comment the print if you want the probability of every knows faces (between 0 and 1)
            #the closer the value is to 0, the greater the probability that it is this person
            #print(face_distance)


            best_match_index = np.argmin(face_distance)
            
            if matches[best_match_index]:
                name = known_faces_names[best_match_index]
                
            cv2.rectangle(frame, (left,top),(right,bottom),(117,50,255),2)
            cv2.rectangle(frame,(left,bottom-35),(right,bottom),(117,50,255),cv2.FILLED)
            font= cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, name,(left+6,bottom-6), font, 0.5,(255,255,255),2)
            
            cv2.imshow('Cam',frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

        input("Press enter to continue...")

    if (answer=='q'):
        break

video_capture.release()
cv2.destroyAllWindows()    


