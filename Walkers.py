import cv2


# Create our body classifier
body_cascade=cv2.CascadeClassifier('/home/galactus/Code/Projects Py/PRO-C106/PRO-106-ProjectTemplate/haarcascade_fullbody.xml')

# Initiate video capture for video file
cap = cv2.VideoCapture('/home/galactus/Code/Projects Py/PRO-C106/PRO-106-ProjectTemplate/walking.avi')

# Loop once video is successfully loaded
while True:
    
    # Read first frame
    ret, frame = cap.read()

    #Convert Each Frame into Grayscale
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # Pass frame to our body classifier
    people=body_cascade.detectMultiScale(gray)
    
    # Extract bounding boxes for any bodies identified
    for (x,y,w,h) in people:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)

    cv2.imshow("Footage",frame)
    

    if cv2.waitKey(1) == 32: #32 is the Space Key
        break

cap.release()
cv2.destroyAllWindows()
