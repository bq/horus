
import cv2

import hardware.uvc_capture as uvc

cv2.namedWindow("w1", cv2.CV_WINDOW_AUTOSIZE)
camera_index = 0




cap= uvc.autoCreateCapture(0,(1280,960));
#cap.controls['UVCC_REQ_EXPOSURE_AUTOMODE'].set_val(8);
#cap.controls['UVCC_REQ_EXPOSURE_AUTOPRIO'].set_val(0);
#capture = cv.CaptureFromCAM(camera_index)

def repeat():
    global capture #declare as globals since we are assigning to them now
    global camera_index
    frame = cap.get_frame()
    
    image=frame.img
    
    #image=cv.resize(image,(1280,960))
    
    if image is None:
        logger.error("Could not retrieve image from capture")
        cap.close()
        return

    image=cv2.resize(image, cap.get_size())
    image = cv2.transpose(image)
    image = cv2.flip(image, 1)

    cv2.imshow("w1", image)
        
    ##print frame.img
    #
    
    #cv.ShowImage("w1", frame)
    #c = cv2.waitKey(10)
    #
    #if (c!=-1):
    #    print c
    #    if (chr(c)=='+'):
    #        cap.controls['UVCC_REQ_EXPOSURE_ABS'].set_val(1000)
    #        
    #    if (chr(c)=='-'):
    #        cap.controls['UVCC_REQ_EXPOSURE_ABS'].set_val(50)
    #    
    #    print ', '.join("%s: %s" % item for item in vars(cap.controls['UVCC_REQ_EXPOSURE_ABS']).items())
    #    
    #    
    #    print "Exposure {0}".format(cap.controls['UVCC_REQ_EXPOSURE_ABS'].get_val())
    
    #if(c=="n"): #in "n" key is pressed while the popup window is in focus
    #    camera_index += 1 #try the next camera index
    #    capture = cv.CaptureFromCAM(camera_index)
    #    if not capture: #if the next camera index didn't work, reset to 0.
    #        camera_index = 0
    #        capture = cv.CaptureFromCAM(camera_index)

while True:
    repeat()