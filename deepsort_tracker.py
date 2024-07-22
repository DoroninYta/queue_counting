from deep_sort.deep_sort.tracker import Tracker as DeepSortTracker
from deep_sort.tools import generate_detections as gdet
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
import numpy as np


class Tracker:
    tracker = None
    encoder = None
    tracks = None

    def __init__(self):
        max_cosine_distance = 0.4
        nn_budget = None

        encoder_model_filename = 'mars-small128.pb'

        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = DeepSortTracker(metric, max_iou_distance=0.7, max_age=30, n_init=8)

        '''Когда max_iou_distance установлен на более высокое значение (ближе к 1), трекер будет более терпимым
        к рамкам, которые частично перекрываются, и это может привести к тому, что более широкий диапазон рамок будет 
        считаться частью одного и того же объекта. Это может привести к объединению нескольких объектов в один.

        Когда max_iou_distance установлен на более низкое значение (ближе к 0), трекер становится более строгим и
        требовательным к тому, чтобы две рамки имели более высокий уровень перекрытия для того, чтобы они были
        считаны как один объект. Это может привести к тому, что объекты будут более четко разделены.'''

        self.encoder = gdet.create_box_encoder(encoder_model_filename, batch_size=1)
        self.max_id = 0

    def update(self, frame, detections):

        if len(detections) == 0:
            self.tracker.predict()
            self.tracker.update([])  
            self.update_tracks()
            return

        bboxes = np.asarray([d[:-1] for d in detections])
        bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, 0:2]
        scores = [d[-1] for d in detections]

        features = self.encoder(frame, bboxes)

        dets = []
        for bbox_id, bbox in enumerate(bboxes):
            dets.append(Detection(bbox, scores[bbox_id], features[bbox_id]))

        self.tracker.predict()
        self.tracker.update(dets)
        self.update_tracks()

    def update_tracks(self):
        tracks = []
        
        for track in self.tracker.tracks:
            # print("tracker_state", track.state, "id", track.track_id)
            if not track.is_confirmed() or track.time_since_update > 1:
                
                continue
            bbox = track.to_tlbr()

            id = track.track_id

            tracks.append(Track(id, bbox))

        self.tracks = tracks
        # print(self.tracks)




class Track:
    track_id = None
    bbox = None
    # history = {}
    # keys = []

    def __init__(self, id, bbox):
        self.track_id = id
        self.bbox = bbox
        # self.history.append((int(bbox[0]), int(bbox[1])))
        # print("its bbox", bbox )
        # self.array_append(bbox, id)

    def array_append(self, arr_elem, id): #make how to delete old track and theirs history
        print(arr_elem)
        arr_elem = self.box_head(arr_elem)
        if id in self.keys:
            # print("in keys")
            self.history[id].append(arr_elem)
            # self.dict[id] += [arr_elem]

        else:
            self.history[id] = [arr_elem]
            self.keys.append(id)


    def box_head(self, box):
        return (int((box[0] + box[2])//2), int((box[1] +30))) 
