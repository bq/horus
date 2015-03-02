import cv

import hardware.uvc_capture as uvc
from hardware.uvc_capture.mac_video import raw

cv.NamedWindow("w1", cv.CV_WINDOW_AUTOSIZE)
camera_index = 0




cap= uvc.autoCreateCapture(0);
cap.controls['UVCC_REQ_EXPOSURE_AUTOMODE'].set_val(1);
cap.controls['UVCC_REQ_EXPOSURE_AUTOPRIO'].set_val(0);

print "Exposure ABS {0}".format(cap.controls['UVCC_REQ_EXPOSURE_ABS'].get_val())
cap.controls['UVCC_REQ_EXPOSURE_ABS'].set_val(1000)

#print cap.controls['UVCC_REQ_BRIGHTNESS_ABS'].set_val(65000)


#print cap.controls['UVCC_REQ_BRIGHTNESS_ABS'].get_val()



print """---\n"""
for control in cap.controls.keys():
    print control
    print """\n\t""".join("%s: %s" % item for item in vars(cap.controls[control]).items())
    print """\n"""

def repeat():
    global cap #declare as globals since we are assigning to them now
    # global camera_index

    cap.controls['UVCC_REQ_BRIGHTNESS_ABS'].set_val(0)
    frame = cap.get_frame()
    
    if frame.img is None:
        logger.error("Could not retrieve image from capture")
        cap.close()
        return
    
    #print frame.img
    
    cv.ShowImage("w1", cv.fromarray(frame.img))
    #cv.ShowImage("w1", frame)
    c = cv.WaitKey(10)
    
    if (c!=-1):
        if (chr(c)=='+'):
            cap.controls['UVCC_REQ_EXPOSURE_ABS'].set_val(1000)
            
        if (chr(c)=='-'):
            cap.controls['UVCC_REQ_EXPOSURE_ABS'].set_val(50)
        
        #print ', '.join("%s: %s" % item for item in vars(cap.controls['UVCC_REQ_EXPOSURE_ABS']).items())
        
        
        #print "Exposure {0}".format(cap.controls['UVCC_REQ_EXPOSURE_ABS'].get_val())
    
    #if(c=="n"): #in "n" key is pressed while the popup window is in focus
    #    camera_index += 1 #try the next camera index
    #    capture = cv.CaptureFromCAM(camera_index)
    #    if not capture: #if the next camera index didn't work, reset to 0.
    #        camera_index = 0
    #        capture = cv.CaptureFromCAM(camera_index)

while True:
    repeat()