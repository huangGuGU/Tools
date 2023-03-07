import cv2
import numpy
import time

# class CaptureManager(object):
#     def __int__(self, capture, previewWindowManager=None, shouldMirrorPreview=False):
#         self.previewWindowManager = previewWindowManager
#         self.shouldMirrorPreview = shouldMirrorPreview
#         self._capture = capture
#         self._channel = 0
#         self._enteredFrame = False
#         self._frame = None
#         self._imageFilename = None
#         self._videoFilename = None
#         self._videoEncoding = None
#         self._videoWriter = None
#         self._startTime = None
#         self._framesElapsed = None
#         self._fpsEstimate = None
#
#     @property
#     def channel(self):
#         return self._channel
#
#     @channel.setter
#     def channel(self, value):
#         if self._channel != value:
#             self._channel = value
#             self._frame = None
#
#     @property
#     def frame(self):
#         if self._enteredFrame and self._frame is None:
#             _, self._frame = self._capture.retrieve()
#         return self._frame
#
#     @property
#     def isWritingImage(self):
#         return self._imageFilename is not None
#
#     @property
#     def isWritingVideo(self):
#         return self._videoFilename is not None
#
#     def enterFrame(self):
#
#         assert not self._enteredFrame, \
#             'previous enterFrame() had no matching exitFrame()'
#
#         if self._capture is not None:
#             self._enteredFrame = self._capture.grab()
#
#     def exitFrame(self):
#
#         if self.frame is None:
#             self.enteredFrame = False
#             return
#
#         if self._framesElapsed == 0:
#             self._startTime = time.time()
#         else:
#             timeElapsed = time.time() - self._startTime
#
#         self._fpsEstimate = self._framesElapsed / timeElapsed
#         self._framesElapsed += 1


# cameraCapture = cv2.VideoCapture(0)
# fps = 30
# size = (int(cameraCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
#         int(cameraCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
#
# videoWriter = cv2.VideoWriter('/Users/hzh/Desktop/x.avi',
#                               cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
#
# success, frame = cameraCapture.read()
# numFramesRemaining = 5 * fps - 1
# while success and numFramesRemaining > 0:
#     videoWriter.write(frame)
#     success, fame = cameraCapture.read()
#     numFramesRemaining -= 1
# cameraCapture.release()



