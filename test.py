import cv2
window_name = 'image'
im = cv2.imread('opencv_frame.png')
im = cv2.resize(im,(96,96))
cv2.imshow(window_name,im)
cv2.waitKey(0) 
  
#closing all open windows 
cv2.destroyAllWindows() 