import cv2
# Create a cascadeclassifier object
face_cascade = cv2.CascadeClassifier('C:\\Users\\ADMIN\\Anaconda3\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')
# Reading the image as it is
img = cv2.imread("C:\\Kat\\Tuyen\\Project\\Facerecognition\\squad.jpg", 1)
# Reading the image as gray scale image
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Search the co-ordinates of the image
faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30))
print(type(faces))
print(faces)

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces found", img)

cv2.waitKey()