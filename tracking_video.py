import os
import random
import cv2
import numpy as np
from time import time  
from ultralytics import YOLO
from deepsort_tracker import Tracker






# таски на завтра
"""
ОБЯЗ НА ЗАВТРА: СЧЕТЧИК ВХОД ВЫХОД
определиться, как понимать когда человек выходит
начать интегрировать нейронку с детекцией по головам 

потестить счетчик

"""







def area_choosing(image, point_roi, points_rect_crop):
    """
    this funtion need for making Region of interests, original image cropping for making less deteckions
    """
        
    pt_area_1 = points_rect_crop[0]
    pt_area_2 = points_rect_crop[1]

    cv2.polylines(image, [point_roi], isClosed = True, color = (255, 0, 255), thickness = 4) #draws area where people entering and exiting
    cv2.rectangle(image, pt_area_1 ,pt_area_2, (255,255,255), 2)  #drawing area, which is giving to YOLO
    img_crop = image[pt_area_1[1]:pt_area_2[1], pt_area_1[0]:pt_area_2[0]].copy()

    return img_crop


pt_area_1_pers = (0, 268)
pt_area_2_pers = (2000, 1080)
pt1_pers = (200, 500)
pt2_pers = (800, 408)
pt3_pers = (1649, 785)      
pt4_pers = (412, 1123)
points = np.array([pt1_pers, pt2_pers, pt3_pers, pt4_pers],np.int32)

video_vid_single_pers = [points, [pt_area_1_pers, pt_area_2_pers]]

# image_crop = area_choosing(img, points, [pt_area_1, pt_area_2])

def box_point(box):
    return (int((box[0] + box[2])//2), int((box[1] +50)))


def cos_angle(a, b):
    # print(f"a = {a}, b = {b}")
    return np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)


def append_history(history, arr_elem, id): #make how to delete old track and theirs history
        
        arr_elem = box_point(arr_elem)
        # print("element append in history", arr_elem)
        if id in history.keys(): #подумать можно как этот момент ускорить, мне это не нравится 
            # print("in keys")
            history[id].append(arr_elem)
            # self.dict[id] += [arr_elem]
            return 0

        else:
            history[id] = [arr_elem]
            return 1
        

def distance_point_to_line(point, line_point1, line_point2):
    """
    Найти расстояние от точки до прямой, заданной двумя точками
    :param point: Точка (x0, y0)
    :param line_point1: Первая точка прямой (x1, y1)
    :param line_point2: Вторая точка прямой (x2, y2)
    :return: Расстояние от точки до прямой
    """
    x0, y0 = point
    x1, y1 = line_point1
    x2, y2 = line_point2

    numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    denominator = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    distance = numerator / denominator

    return distance


def distance_point_to_point(pt1, pt2):
    dist = np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
    return dist

def makes_roi_bigger(point, value = 50):
    points = point.copy()
    point1 = points[0]
    point2 = points[1]
    point3 = points[2]
    point4 = points[3]

    point1[0] -= value
    point1[1] -= value
    point2[0] += value
    point2[1] -= value
    point3[0] += value*2
    point3[1] += value
    point4[0] -= value
    point4[1] += value

    return np.array([point1, point2, point3, point4],np.int32)




def find_vector_speed(array_point : list):
    
    """
    lookking at latest or less 30 points
    return average direction - what you reaaly need, array of point to point directions, indexes of data for previos (<--) value)
    """
    
    directions = []
    array_indexes = []
    arr_len = len(array_point)

    start_index = 2
    step  = 2 
    if len(array_point) < 2:
        return np.array([1, 1]), 1, 1
    if arr_len > 30:
        start_index = arr_len - 30

    elif arr_len < 8:
        start_index = 1
        step = 1

    
    for i in range(start_index, arr_len, step):
        start_point = np.array(array_point[i - step])
        end_point = np.array(array_point[i])
        array_indexes.append(i - step)
        direction_vector = end_point - start_point
        # direction_vector = direction_vector / np.linalg.norm(direction_vector)
        directions.append(direction_vector)

    average_direction = np.mean(directions, axis = 0)
    if (average_direction == (0, 0)).all():
        print("there zero vector")
        average_direction = np.array([1,1])

    return  average_direction,directions,array_indexes


def main_func(video_path : str, points, points_area, only_head_detecting = False):
    if only_head_detecting:
        model = YOLO("../head_detection.pt")
        # model = YOLO("../crowdhuman_yolov5m.pt")
    else:
        model = YOLO("yolov8m.pt")
    model.fuse()
    tracker = Tracker()
    history = {}
    array_id_inside = []
    count_attration = 0
    count_roi = 0
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(10)]
    norm_vect = (-10, 4)
    frame_number = 0
    max_angle = 0.7
    point_of_entering = (357, 787)
    roi_for_tracking = makes_roi_bigger(points)
    # roi_for_tracking = points

    detection_threshold = 0.3

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    #Import only if not previously imported
    
    
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    writer = cv2.VideoWriter('video_tracking_save.avi',fourcc,25, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    
    file = open("dump_file.txt", "w")
    
    #next mine
    while True:
        frame_number += 1
        start_time = time()

        ret, frame = cap.read()
        # if frame_number < 5250:
            # continue
        if not ret:
            break  # Exit the loop if there's no frame
        frame_crop = area_choosing(frame, points, points_area)

        results = model.predict(frame_crop, conf = 0.2, iou = 0.3, save = False,imgsz = 1280, classes = [0], verbose=False)
        cv2.polylines(frame, [points], isClosed = True, color = (255, 0, 255), thickness = 4)
        cv2.polylines(frame, [roi_for_tracking], isClosed = True, color = (255, 111, 255), thickness = 4)

        for result in results:
            detections = []
            for r in result.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = r
                x1 = int(x1 + points_area[0][0])
                x2 = int(x2 + points_area[0][0])
                y1 = int(y1 + points_area[0][1])
                y2 = int(y2 + points_area[0][1])
                
                class_id = int(class_id)

                roi_distance = cv2.pointPolygonTest(roi_for_tracking, box_point([x1, y1, x2, y2]), True)

                cv2.circle(frame,(box_point([x1, y1, x2, y2])), 5,  (0,0,255), -1)           
                if score > detection_threshold and roi_distance >= 0:
                    detections.append([x1, y1, x2, y2, score])
                
                elif score > detection_threshold and roi_distance < 0:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 3)
                    cv2.putText(frame, "Not tracking", (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            tracker.update(frame, detections)

            # print("state", tracker.tracker.tracks[1].time_since_update)
            array_id_tracking = [tr.track_id for tr in tracker.tracker.tracks]
            # print("array_id, working", ar)
            
            font = cv2.FONT_HERSHEY_COMPLEX_SMALL
            cv2.putText(frame, f"{array_id_tracking}", (2000, 1200),font, 2, (0, 0, 255),4, cv2.LINE_AA)

            
            for track in tracker.tracks:
                
                
                bbox = track.bbox
                x1, y1, x2, y2 = bbox
                track_id = track.track_id

                is_new = append_history(history, bbox, track_id)
                # if is_new:
                #     array_id_inside.append(track_id)

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), colors[track_id % len(colors)], 3)
                cv2.putText(frame, "ID: " + str(track_id), (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                #drowing track
                for box in history[track_id]:
                    cv2.circle(frame,(box), 5,  colors[track_id % len(colors)],-1)  
                
                #finding and drawing speed vector
                # !TODO
                vector_speed, _, _ = find_vector_speed(history[track_id])
                angle_between_lines = cos_angle(vector_speed, norm_vect) #return it to if later
                # MAKE SPECIAL FUNCTION HERE LATER
                head_pos = box_point(bbox)
                distance_barricade = cv2.pointPolygonTest(points, head_pos, measureDist = True)
                
                pppt2 = (head_pos * vector_speed / 10 + head_pos).astype(int)
                
                cv2.arrowedLine(frame, head_pos, pppt2, colors[track_id % len(colors)], 3, 5, 0, 0.5)
                dist = distance_point_to_point(point_of_entering, history[track_id][0])

                file.write(f"\n---------\nTrack_id = {track_id}, distance_barricade = {distance_barricade}, \ndistance to edge line = {dist} len(hist) = {len(history[track_id])}, angle = {angle_between_lines}\nframe number = {frame_number}")

                #looking for new people in roi
                if track_id not in array_id_inside and len(history[track_id]) >6:        #append append array_id_inside and add to here, change dist to 30 maybe
                    #here we are checking new peoples in the roi
                   
                    array_id_inside.append(track_id)
                    # vector_speed, _, _ = find_vector_speed(history[track_id])
                    
                    cv2.arrowedLine(frame, head_pos, (head_pos * vector_speed / 30 + head_pos).astype(int), colors[track_id % len(colors)], 3, 5, 0, 0.5)

                    if  angle_between_lines < - max_angle:
                        
                        if dist < 200: 
                            # situation, when people is going to attraction
                            print(f"people with track number {track_id} coming in ROI and going from attraction with angle = {angle_between_lines}, distance to line is {dist}")
                            print("count--")
                            count_attration-=1
                        
                    
            for track_id in array_id_inside:
                
                if  track_id in array_id_inside and track_id not in array_id_tracking:

                    if len(history[track_id]) > 5:
                        array_id_inside.remove(track_id)
                        vector_speed, _, _ = find_vector_speed(history[track_id])
                        cv2.arrowedLine(frame, head_pos, (head_pos * vector_speed / 10 + head_pos).astype(int), colors[track_id % len(colors)], 3, 5, 0, 0.5)

                        angle_between_lines = cos_angle(vector_speed, norm_vect)
                        print(f"checking about exit: id {track_id}, angle = {angle_between_lines}")

                        if  angle_between_lines > max_angle or distance_point_to_point(point_of_entering, history[track_id][-1]) < 200 :
                            # situation, when people is going to attraction
                            print(f"people with track number {track_id} getting out from the ROI and going to attraction,angle = ")
                            print("count++")
                            count_attration+=1
                        else:
                            print(f"not today, angle = {angle_between_lines}, id = {track_id}")

                    else:
                        array_id_inside.remove(track_id) 
                        

                # print("\n\n---------------------\n")

            cv2.putText(frame, f"{array_id_inside}", (2000, 1000),font, 2, (0, 0, 255),4, cv2.LINE_AA)
            cv2.putText(frame, f"Count in ROI {len(array_id_inside)}", (1000, 1100),font, 2, (255, 0, 255),4, cv2.LINE_AA)
            cv2.putText(frame, f"Count in attraction {count_attration}", (1000, 1200),font, 2, (255, 0, 255),4, cv2.LINE_AA)
            cv2.putText(frame, f"frame_number = {frame_number}", (1000, 1400),font, 2, (0, 0, 0),4, cv2.LINE_AA)


                        




            




                

        end_time = time()
        fps = 1/np.round(end_time - start_time, 2)
            
        cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
        cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Tracking", 1600, 1100)

        cv2.imshow("Tracking", frame)
        writer.write(frame)


        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    writer.release()
    cap.release()
    cv2.destroyAllWindows()
    file.close()

main_func("../../video/main_vid.avi", points, [pt_area_1_pers, pt_area_2_pers],True)



