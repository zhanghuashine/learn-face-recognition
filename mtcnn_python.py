
import caffe
import cv2
import os
import numpy as np
import math
import time

class MtcnnDetect(object):
    MIN_FACE_SIZE = 40
    PNET_CELL_SIZE = 12
    BACH_SIZE = 128
    PNET_STRIDE = 2
    FACTOR = 0.709  # 1 / math.sqrt(2)
    NMS_THREAD = 0.7
    NET_THREADS = [0.95, 0.95, 0.95]

    def __init__(self, nets_dir, trained_dir, img_size=(640, 480), detect_model='GPU'):
        caffe.set_device(0)
        if detect_model == 'GPU':
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()
        self._pnet = caffe.Net(os.path.join(nets_dir, 'det1.prototxt'), os.path.join(trained_dir, 'det1.caffemodel'),
                               caffe.TEST)
        self._rnet = caffe.Net(os.path.join(nets_dir, 'det2.prototxt'), os.path.join(trained_dir, 'det2.caffemodel'),
                               caffe.TEST)
        self._onet = caffe.Net(os.path.join(nets_dir, 'det3.prototxt'), os.path.join(trained_dir, 'det3.caffemodel'),
                               caffe.TEST)
        # pre process
        self._imgSize = img_size
        self.__scales = []
        self.__bboxes = []

    def IsGetReady(self):
        self.__ImagePyramid(self._imgSize)
        if len(self.__scales) == 0:
            return False
        else:
            return True

    def GetDetResult(self, img, upsample=1):
        img_norm = (img - 127.5) * 0.0078125
        # start = time.clock()
        self._GoPnet(img_norm)
        # start = time.clock()
        # print('PNet time: ' + str(round(end - start, 3)))
        self._GoRnet(img_norm)
        # end = time.clock()
        # print('RNet time: ' + str(round(end - start, 3)))
        # start = time.clock()
        ret = self._GoOnet(img_norm)
        # end = time.clock()
        # print('ONet time: ' + str(round(end - start, 3)))

        return ret

    def _GoPnet(self, img):
        rows, cols, channels = img.shape
        self.__bboxes = []  # list clear, if have the better method?

        for scale in self.__scales:
            h, w = int(math.ceil(rows * scale)), int(math.ceil(cols * scale))
            if h < 12 or w < 12:
                continue
            img_scale = cv2.resize(img, (w, h))
            input_data = np.zeros((1, h, w, channels))
            input_data[0, ...] = img_scale
            input_data = input_data.transpose(0, 3, 1, 2)

            self._pnet.blobs['data'].reshape(1, channels, h, w)
            self._pnet.blobs['data'].data[...] = input_data
            self._pnet.forward()

            scores = self._pnet.blobs['prob1'].data[0]
            regs = self._pnet.blobs['conv4-2'].data[0]
            # start = time.clock()
            bbox = self.__GenerateGoodBBox((scores, regs), self.NET_THREADS[0], scale)  # must have better method!
            # end = time.clock()
            # print('__GenerateGoodBBox time: ' + str(round(end - start, 3)))
            if len(bbox) != 0:
                self.__bboxes.extend(bbox)

        self.__bboxes = np.array(self.__bboxes)
        self.__NMS(self.NMS_THREAD)
        self.__RegBoxes()
        self.__PadBboxesSquare((cols, rows))


    def _GoRnet(self, img):
        if len(self.__bboxes) == 0:
            return

        rows, cols, channels = img.shape
        x1 = self.__bboxes[:, 1]
        y1 = self.__bboxes[:, 2]
        x2 = self.__bboxes[:, 3]
        y2 = self.__bboxes[:, 4]

        length = len(self.__bboxes)  # ONet bboxes' num
        num = int(math.floor(length * 1.0 / self.BACH_SIZE))
        restnum = length - num * self.BACH_SIZE
        for i in range(num):
            self._rnet.blobs['data'].reshape(self.BACH_SIZE, 3, 24, 24)
            input_data = np.zeros((self.BACH_SIZE, 24, 24, 3))
            for j in range(self.BACH_SIZE):
                roi = img[int(y1[self.BACH_SIZE * i + j]):int(y2[self.BACH_SIZE * i + j]),
                                    int(x1[self.BACH_SIZE * i + j]):int(x2[self.BACH_SIZE * i + j])]
                roi = cv2.resize(roi, (24, 24))
                input_data[j, :, :, :] = roi
            input_data = input_data.transpose(0, 3, 1, 2)
            self._rnet.blobs['data'].data[...] = input_data
            self._rnet.forward()
            scores = self._rnet.blobs['prob1'].data[...]
            regs = self._rnet.blobs['conv5-2'].data[...]

            self.__bboxes[i * self.BACH_SIZE: (i + 1) * self.BACH_SIZE, 0] = scores[:, 1]
            self.__bboxes[i * self.BACH_SIZE: (i + 1) * self.BACH_SIZE, 5:] = regs

        if restnum > 0:
            self._rnet.blobs['data'].reshape(restnum, 3, 24, 24)
            input_data = np.zeros((restnum, 24, 24, 3))
            for i in range(restnum):
                # print int(x1[i]), int(x2[i]), int(y1[i]), int(y2[i])
                roi = img[int(y1[i]):int(y2[i]), int(x1[i]):int(x2[i])]
                roi1 = cv2.resize(roi, (24, 24))
                input_data[i, :, :, :] = roi1
            input_data = input_data.transpose(0, 3, 1, 2)
            self._rnet.blobs['data'].data[...] = input_data
            self._rnet.forward()
            scores = self._rnet.blobs['prob1'].data[...]
            regs = self._rnet.blobs['conv5-2'].data[...]
            self.__bboxes[num*self.BACH_SIZE:, 0] = scores[:, 1]
            self.__bboxes[num*self.BACH_SIZE:, 5:] = regs
            self.__bboxes = np.delete(self.__bboxes, np.where(self.__bboxes[num*self.BACH_SIZE:, 0] < self.NET_THREADS[1]), axis=0)

        self.__NMS(self.NMS_THREAD)
        self.__RegBoxes()
        self.__PadBboxesSquare((cols, rows))


    # include: NMS, Regression, Padding ---- > good bboxes
    def _GoOnet(self, img):
        if len(self.__bboxes) == 0:
            return []

        length = len(self.__bboxes)
        num = int(math.floor(length * 1.0 / self.BACH_SIZE))
        restnum = length - num * self.BACH_SIZE

        x1 = self.__bboxes[:, 1]
        y1 = self.__bboxes[:, 2]
        x2 = self.__bboxes[:, 3]
        y2 = self.__bboxes[:, 4]

        for i in range(num):
            self._onet.blobs['data'].reshape(self.BACH_SIZE, 3, 48, 48)
            input_data = np.zeros((self.BACH_SIZE, 48, 48, 3))
            for j in range(self.BACH_SIZE):
                roi = img[int(y1[self.BACH_SIZE * i + j]):int(y2[self.BACH_SIZE * i + j]),
                                    int(x1[self.BACH_SIZE * i + j]):int(x2[self.BACH_SIZE * i + j])]
                roi1 = cv2.resize(roi, (48, 48))
                input_data[j, :, :, :] = roi1
            input_data = input_data.transpose(0, 3, 1, 2)
            self._onet.blobs['data'].data[...] = input_data
            self._onet.forward()
            scores = self._onet.blobs['prob1'].data[...]
            regs = self._onet.blobs['conv6-2'].data[...]
            # landmarks
            landmarks = self._onet.blobs["conv6-3"].data[...]
            landmarks = np.array(landmarks)

            w = self.__bboxes[i * self.BACH_SIZE:, 3] - self.__bboxes[i * self.BACH_SIZE:, 1]
            h = self.__bboxes[i * self.BACH_SIZE:, 4] - self.__bboxes[i * self.BACH_SIZE:, 2]
            for j in range(5):
                landmarks[:, 2 * j] = landmarks[:, 2 * j] * w + self.__bboxes[i * self.BACH_SIZE:, 1]
                landmarks[:, 2 * j + 1] = landmarks[:, 2 * j + 1] * h + self.__bboxes[i * self.BACH_SIZE:, 2]

            self.__bboxes = np.append(self.__bboxes[i*self.BACH_SIZE:], landmarks, axis=1)

            self.__bboxes[i * self.BACH_SIZE: (i + 1) * self.BACH_SIZE, 0] = scores[:, 1]
            self.__bboxes[i * self.BACH_SIZE: (i + 1) * self.BACH_SIZE, 5:9] = regs  # landmark
            # self.__bboxes[i * self.BACH_SIZE: (i + 1) * self.BACH_SIZE, 5:] = regs

            self.__bboxes[i * self.BACH_SIZE: (i + 1) * self.BACH_SIZE] = np.delete(self.__bboxes[i * self.BACH_SIZE: (i + 1) * self.BACH_SIZE],
                                                                                    np.where(self.__bboxes[i * self.BACH_SIZE: (i + 1) * self.BACH_SIZE, 0] < self.NET_THREADS[2]), axis=0)


        if restnum > 0:
            self._onet.blobs['data'].reshape(restnum, 3, 48, 48)
            input_data = np.zeros((restnum, 48, 48, 3))
            for i in range(restnum):
                # print int(x1[i]), int(x2[i]), int(y1[i]), int(y2[i])
                roi = img[int(y1[i]):int(y2[i]), int(x1[i]):int(x2[i])]
                roi1 = cv2.resize(roi, (48, 48))
                # cv2.imshow('', roi)
                # cv2.waitKey(0)
                input_data[i, :, :, :] = roi1
            input_data = input_data.transpose(0, 3, 1, 2)
            self._onet.blobs['data'].data[...] = input_data
            self._onet.forward()
            scores = self._onet.blobs['prob1'].data[...]
            regs = self._onet.blobs['conv6-2'].data[...]

            # landmarks
            landmarks = self._onet.blobs["conv6-3"].data[...]
            landmarks = np.array(landmarks)

            w = self.__bboxes[num*self.BACH_SIZE:, 3] - self.__bboxes[num*self.BACH_SIZE:, 1]
            h = self.__bboxes[num*self.BACH_SIZE:, 4] - self.__bboxes[num*self.BACH_SIZE:, 2]
            for i in range(5):
                landmarks[:, 2 * i] = landmarks[:, 2 * i] * w + self.__bboxes[num*self.BACH_SIZE:, 1]
                landmarks[:, 2 * i + 1] = landmarks[:, 2 * i + 1] * h + self.__bboxes[num*self.BACH_SIZE:, 2]

            self.__bboxes = np.append(self.__bboxes[num*self.BACH_SIZE:], landmarks, axis=1)  # extend self.__bboxex dims for include landmarks
            self.__bboxes[num*self.BACH_SIZE:, 0] = scores[:, 1]
            self.__bboxes[num*self.BACH_SIZE:, 5:9] = regs  # include landmark
            # self.__bboxes[num*self.BACH_SIZE:, 5:] = regs  # not include landmark
            self.__bboxes = np.delete(self.__bboxes, np.where(self.__bboxes[num*self.BACH_SIZE:, 0] < self.NET_THREADS[2]), axis=0)

        self.__RegBoxes()
        self.__NMS(self.NMS_THREAD, model='m')
        self.__PadBboxes((img.shape[1], img.shape[0]))

        return self.__bboxes

    # private
    def __ImagePyramid(self, img_size):
        h, w = img_size
        scale = 12.0 / self.MIN_FACE_SIZE
        min_side = min(w, h) * scale
        while min_side >= 12.0:
            self.__scales.append(scale)
            min_side *= self.FACTOR
            scale *= self.FACTOR
        self.__scales.pop(0)


    # return: []---->[score, left, top, right, bottom, regRight, ...]
    def __GenerateGoodBBox(self, pnet_out, score_thread, scale):
        scores = pnet_out[0]
        regs = pnet_out[1]
        c, h, w = scores.shape
        bboxes = []
        # change to numpy for speeding up!
        for i in range(h):
            for j in range(w):
                if scores[1][i][j] > score_thread:
                    bbox = [float(scores[1][i][j])]
                    i_ = i * self.PNET_STRIDE
                    j_ = j * self.PNET_STRIDE
                    bbox.append(j_ / scale)   # left
                    bbox.append(i_ / scale)  # top
                    bbox.append((self.PNET_CELL_SIZE - 1.0 + j_) / scale)   # right
                    bbox.append((self.PNET_CELL_SIZE - 1.0 + i_) / scale)   # bottom

                    for k in range(regs.shape[0]):
                        bbox.append(float(regs[k][i][j]))
                    bboxes.append(bbox)
        return bboxes


    def __NMS(self, overlapthread, model='u'):
        # if pnet doesn't propose  bbox, just return
        if len(self.__bboxes) == 0:
            return

        if self.__bboxes.dtype.kind == 'i':
            self.__bboxes = self.__bboxes.astype('float')

        pick = []  # index list

        score = self.__bboxes[:, 0]

        x1 = self.__bboxes[:, 1]
        y1 = self.__bboxes[:, 2]
        x2 = self.__bboxes[:, 3]
        y2 = self.__bboxes[:, 4]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(score)    # From small to large, return index

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            # compute the ratio of overlap
            area_test = w * h
            if model == 'u':
                overlap = (w * h) / (area[idxs[:last]] + area[i] - area_test)  # (area_intersect / area1) with author has a little difference
            else:
                overlap = (w * h) / area[idxs[:last]]

            # delete all indexes from the index list that have
            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapthread)[0])), axis=0)

        self.__bboxes = self.__bboxes[pick]


    def __RegBoxes(self):
        if len(self.__bboxes) == 0:
            return

        w = self.__bboxes[:, 3] - self.__bboxes[:, 1] + 1
        h = self.__bboxes[:, 4] - self.__bboxes[:, 2] + 1
        self.__bboxes[:, 1] += self.__bboxes[:, 5] * w
        self.__bboxes[:, 2] += self.__bboxes[:, 6] * h
        self.__bboxes[:, 3] += self.__bboxes[:, 7] * w
        self.__bboxes[:, 4] += self.__bboxes[:, 8] * h

    # other tricks
    def __PadBboxes(self, img_size):
        self.__bboxes[:, 1] = np.maximum(self.__bboxes[:, 1], 0.0)
        self.__bboxes[:, 2] = np.maximum(self.__bboxes[:, 2], 0.0)
        self.__bboxes[:, 3] = np.minimum(self.__bboxes[:, 3], img_size[0] - 1.0)
        self.__bboxes[:, 4] = np.minimum(self.__bboxes[:, 4], img_size[1] - 1.0)

    def __PadBboxesSquare(self, img_size):
        if len(self.__bboxes) == 0:
            return

        w = self.__bboxes[:, 3] - self.__bboxes[:, 1] + 1
        h = self.__bboxes[:, 4] - self.__bboxes[:, 2] + 1
        side = np.maximum(w, h)

        self.__bboxes[:, 1] = np.around(np.maximum(self.__bboxes[:, 1] + (w - side) * 0.5, 0.))
        self.__bboxes[:, 2] = np.around(np.maximum(self.__bboxes[:, 2] + (h - side) * 0.5, 0.))
        self.__bboxes[:, 3] = np.around(np.minimum(self.__bboxes[:, 1] + side - 1.0, img_size[0] - 1.0))
        self.__bboxes[:, 4] = np.around(np.minimum(self.__bboxes[:, 2] + side - 1.0, img_size[1] - 1.0))



if __name__ == '__main__':
    try:
        mtdtr = MtcnnDetect('./network/mtcnn', './network/mtcnn')
        isReady = mtdtr.IsGetReady()
        video = cv2.VideoCapture(0)
        if video.isOpened(): isopened = True
        else: isopened = False
        if isReady and isopened:
            while True:
                start = time.clock()
                ret, img = video.read()
                result = mtdtr.GetDetResult(img)
                for box in result:
                    cv2.rectangle(img, (int(box[1]), int(box[2])), (int(box[3]), int(box[4])), (0, 255, 0), 2)
                    for i in range(5):
                        cv2.circle(img, (int(box[9 + i*2]), int(box[10 + i*2])), 2, (55, 255, 155), -1)
                    
                end = time.clock()
                tm = end - start
                fps = round(1.0 / tm, 3)
                cv2.putText(img, str(fps), (0, 20), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 1)
                cv2.imshow('', img)
                key = cv2.waitKey(1)
                if key & 255 == 27:
                    video.release()
                    cv2.destroyAllWindows()
                    break
    except Exception as e:
        print e
