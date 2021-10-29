import time
import cv2
import numpy as np
from numpy.linalg import norm
from sklearn.preprocessing import MaxAbsScaler
import pickle
from PIL import Image
import pyttsx3
from multiprocessing.pool import ThreadPool
import threading
import queue
from statistics import mode

# Thread for Text to Speech Engine
class TTSThread(threading.Thread):
    def __init__(self, queue):
        threading.Thread.__init__(self)
        self.queue = queue
        self.daemon = True
        self.start()

    def run(self):
        tts_engine = pyttsx3.init()
        tts_engine.startLoop(False)
        while True:
            if self.queue.empty():
                tts_engine.iterate()
            else:
                data = self.queue.get()
                tts_engine.say(data)
        tts_engine.endLoop()

    
def most_common(List):
    return(mode(List))


def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img


def make_predict(sample, old_digit):
    lr_y = str(lr.predict(sample)[0].ravel())
    knn_y = str(knn.predict(sample)[0].ravel())
    rf_y = str(rf.predict(sample)[0].ravel())
    svc_y = str(svc.predict(sample)[0].ravel())

    # vote for predictions
    digit = most_common([lr_y, knn_y, rf_y, svc_y])

    if digit != old_digit:
        q.put(digit)
    
    return digit, lr_y, knn_y, rf_y, svc_y

def has_moved(i1, i2): 
    pairs = zip(i1.getdata(), i2.getdata())
    if len(i1.getbands()) == 1:
        # for gray-scale jpegs
        dif = sum(abs(p1-p2) for p1,p2 in pairs)
    else:
        dif = sum(abs(c1-c2) for p1,p2 in pairs for c1,c2 in zip(p1,p2))
 
    ncomponents = i1.size[0] * i1.size[1] * 3
    percentage = (dif / 255.0 * 100) / ncomponents

    print(percentage)
    
    if percentage < 1:
        return False
    else:
        return True


# init variables
SZ = 28
old_digit = 0
old_frame = None
async_results = None

# load the models from disk
lr_name = 'joymnist_LR.sav'
knn_name = 'joymnist_KNN.sav'
rf_name = 'joymnist_RF.sav'
svc_name = 'joymnist_SVC.sav'

lr = pickle.load(open(lr_name, 'rb'))
knn = pickle.load(open(knn_name, 'rb'))
rf = pickle.load(open(rf_name, 'rb'))
svc = pickle.load(open(svc_name, 'rb'))

# define a MaxAbsScaler object
scaler = MaxAbsScaler()

# create queue to send commands from the main thread
q = queue.Queue()

# create thread to accept TTS commands
tts_thread = TTSThread(q)

# create a pool of threads for async tasks
pool = ThreadPool(processes=1)

# define a video capture object
vid = cv2.VideoCapture(0)

while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    #diff_frame = cv2.absdiff(old_frame, frame)
    
    # Converting BGR-> Gray and making frame blur
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    bin = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 121, 100)
    #bin = cv2.GaussianBlur(bin, (21, 21), 0)
    bin = cv2.medianBlur(bin, 3)

    contours, heirs = cv2.findContours(bin.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    try:
        heirs = heirs[0]
    except:
        heirs = []

    for cnt, heir in zip(contours, heirs):
        _, _, _, outer_i = heir
        if outer_i >= 0:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if not (128 <= h <= 1024 and w <= 1.2*h):
            continue
        pad = max(h-w, 0)
        x, w = x - (pad // 2), w + pad
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0))
        
        bin_roi = bin[y:,x:][:h,:w]
        m = bin_roi != 0
        if not 0.0 < m.mean() < 0.8:
            continue
        s = 1.5*float(h)/SZ
        m = cv2.moments(bin_roi)
        c1 = np.float32([m['m10'], m['m01']]) / m['m00']
        c0 = np.float32([SZ/2, SZ/2])
        t = c1 - s*c0
        A = np.zeros((2, 3), np.float32)
        A[:,:2] = np.eye(2)*s
        A[:,2] = t
        bin_norm = cv2.warpAffine(bin_roi, A, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
        bin_norm = deskew(bin_norm)
        if x+w+SZ < frame.shape[1] and y+SZ < frame.shape[0]:
            frame[y:,x+w:][:SZ, :SZ] = bin_norm[...,np.newaxis]

        # transform image
        img_pil = Image.fromarray(bin_norm)
        resize = np.array(img_pil.resize((28, 28), Image.ANTIALIAS))
        flatten = (resize.flatten())
        reshape = flatten.reshape(-1,1).T
        sample = scaler.fit_transform(reshape)

        # make predictions
        if old_frame is not None:
            if not has_moved(img_pil, old_frame):
                async_results = pool.apply_async(make_predict, (sample, old_digit,))
                digit, lr_y, knn_y, rf_y, svc_y = async_results.get()
                old_digit = digit


        if async_results is not None:
            cv2.putText(frame, '%s'%digit, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (200, 0, 0), thickness = 1)
            cv2.putText(frame, "Logistic Regression : " + lr_y, (10, 320),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "KNN : " + knn_y, (10, 350),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Random Forest : " + knn_y, (10, 380),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "SVC :  " + svc_y, (10, 410),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        old_frame = img_pil

    # Display the resulting frame
    cv2.imshow('frame', frame)
    #cv2.imshow('bin', bin)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()

# Destroy all the windows
cv2.destroyAllWindows()
