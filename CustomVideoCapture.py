import threading
import cv2
import time

class CustomVideoCapture:
    """
    CustomVideoCapture class:
        This class is designed to be a wrapper for cv2.VideoCapture to streamline
        video feeds by utilizing threading.

        The primary function is that the read() function is run constantly in a separate thread.
        Locks and Events are used to allow this class to be thread-safe.

    Parameters:
        video_source ::: Can be a filepath to a video file, rstp/rtmp/http video feed, or anything the normal cv2.VideoCapture accepts

        capped_fps ::: Set True if using a file-based video. Set to False by default for rstp/rtmp/http feeds, 
        as we want to read frames as soon as they are available.

        framerate ::: Sets the framerate for video feeds, only applied when capped_fps is True.

        restart_on_end ::: Restarts the video from a file in case it ends. True by default.
    """
    
    last_frame = None
    last_ready = None
    lock = threading.Lock()
    stop_event = threading.Event()
    start_event = threading.Event()
    fps = 30
    video_source = None
    capped_fps = False
    restart_on_end = True

    ## ensure capped_fps is False for case of rstp/rtmp url 
    def __init__(self, video_source:str, framerate:int=30, capped_fps:bool=False, restart_on_end:bool=True):
        self.fps : int = framerate
        self.video_source : str = video_source
        self.capped_fps : bool = capped_fps
        self.restart_on_end : bool = restart_on_end
        self.cap : cv2.VideoCapture = cv2.VideoCapture(video_source)
        self.thread : threading.Thread = threading.Thread(target=self.__capture_read_thread__, name="rtsp_read_thread")
        self.thread.daemon = True
        self.thread.start()

    def __capture_read_thread__(self):
        while not self.stop_event.is_set():
            with self.lock:
                self.last_ready, self.last_frame = self.cap.read()
            
            if not self.last_ready:  # if video ended or failed to read frame
                if self.restart_on_end:
                    with self.lock:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # restart video
                else:
                    break
            
            # only wait in case of video files, else keep reading frames without delay to prevent packet drop/burst receive
            if self.capped_fps:
              time.sleep(1 / self.fps)
        return

    def read(self):
        if self.start_event.is_set():
            if (self.last_ready is not None) and (self.last_frame is not None):
                return [self.last_ready, self.last_frame.copy()]
        else:
            if not self.stop_event.is_set():
                self.start_event.set()

        return [False, None]
          
    def isOpened(self):
        return self.cap.isOpened()
    
    def open(self, video_source):
        with self.lock:
            self.cap.open(video_source)

    def release(self):
        self.stop_event.set()
        self.restart_on_end = False
        self.thread.join(2)
        
        with self.lock:
            self.cap.release()
