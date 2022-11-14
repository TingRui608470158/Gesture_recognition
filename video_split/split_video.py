import cv2

cap = cv2.VideoCapture('video7.mp4')

frame_num = 0
while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_bgr = gray.copy()
    gray_bgr =  cv2.cvtColor(gray_bgr, cv2.COLOR_GRAY2RGB)
   
    if frame_num%5 ==0:
        # print(frame_num/3)
        # cv2.imshow('frame', gray_bgr)
        # print("dataset/video1/"+ str(int(frame_num/3)) +'.jpg')
        cv2.imwrite("data/video7_"+ str(int(frame_num/5)) +'.jpg', gray_bgr)
    frame_num = frame_num + 1
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()