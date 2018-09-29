import cv2

class VideoStreamReader:
    def __init__(self, filename, seconds_count, seconds_skip = 0, downscale = 1):
        self.filename = filename
        self.capture = cv2.VideoCapture(filename)
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)) // downscale
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) // downscale
        self.fps = int(self.capture.get(cv2.CAP_PROP_FPS))
        self.seconds_skip = seconds_skip
        self.seconds_count = seconds_count
        self.frame_skip = int(self.fps * seconds_skip)
        self.frame_count = int(self.fps * seconds_count)
        self.frame_no = -1      
        self.downscale = downscale

        for i in range(self.frame_skip):
            self.capture.grab()
            self.frame_no +=1 

    def next_frame(self):
        if self.frame_no >= self.frame_skip + self.frame_count:
            return None
        
        ret, frame = self.capture.read()
        if not ret:
            return None

        self.frame_no +=1 
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        if self.downscale != 1:
            frame = cv2.resize(frame, (0,0), fx=1.0 / self.downscale, fy=1.0 / self.downscale)
        
        return frame

    def release(self):
        self.capture.release()

class VideoStreamWriter:
    def __init__(self, filename, width, height, fps):
        self.filename = filename
        self.width = width
        self.height = height
        self.fps = fps
        self.writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), fps, (width,height))

    def write(self, frame):
        self.writer.write(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))

    def release(self):
        self.writer.release()