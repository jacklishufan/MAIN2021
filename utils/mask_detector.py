from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import os,cv2
import numpy as np
import itertools
def load_predictor(model_path):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 1000
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
    predictor = DefaultPredictor(cfg)
    return predictor

def findIntersection(line1, line2):
        x1,y1,x2,y2 = line1
        x3,y3,x4,y4 = line2
        px= ( (x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) ) 
        py= ( (x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) )
        return (px, py)

def angle(line1):
    x0,y0,x1,y1 = line1
    theta1 = np.arctan2(y1-y0,x1-x0)
    return theta1

def get_lines(poly):
    lines = []
    n = len(poly)
    for i in range(n):
        x0,y0 = poly[i][0]
        if i+1 < n:
            t = i+1
        else:
            t = 0
        x1,y1 = poly[t][0]
        lines.append((x0,y0,x1,y1))
    return lines

def get_line_length(line):
    x0,y0,x1,y1 = line
    return np.linalg.norm([x0-x1,y0-y1])

def orer_cordinates(cordinates):
    x0,y0 = np.mean(cordinates,axis=0)
    lst = [(angle((x0,y0,x,y)),(x,y)) for x,y in cordinates]
    lst = sorted(lst)
    print(lst)
    return [x[1] for x in lst]

def get_intersectios(lines,xb,yb):
    lst = []
    for l1,l2 in itertools.combinations(lines,2):
        if l1 == l2:
            continue
        if abs(angle(l1)-angle(l2))< np.angle(10):
            continue
        px,py = findIntersection(l1,l2)
        lst.append([px,py])
    x_min = -yb * 0.1
    x_max = yb * 1.1
    y_min = -xb * 0.1
    y_max = xb * 1.1
    lst = [(x,y) for x,y in lst if x>=x_min and x<= x_max and y >= y_min and y <= y_max]
    return lst

def autocrop(image, threshold=0):
    """Crops any edges below or equal to threshold

    Crops blank image to 1x1.

    Returns cropped image.

    """
    if len(image.shape) == 3:
        flatImage = np.max(image, 2)
    else:
        flatImage = image
    assert len(flatImage.shape) == 2

    rows = np.where(np.max(flatImage, 0) > threshold)[0]
    if rows.size:
        cols = np.where(np.max(flatImage, 1) > threshold)[0]
        image = image[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]
    else:
        image = image[:1, :1]

    return image

def transform_perspective(im,key_points):
    rows,cols,ch = im.shape

    pts1 = np.float32(key_points)
    offsetSize=max(rows,cols)*2
    ratio=1
    cardH=np.sqrt((pts1[2][0]-pts1[1][0])*(pts1[2][0]-pts1[1][0])+(pts1[2][1]-pts1[1][1])*(pts1[2][1]-pts1[1][1]))
    cardW=ratio*cardH
    pts2 = np.float32([[pts1[0][0],pts1[0][1]], [pts1[0][0]+cardW, pts1[0][1]], [pts1[0][0]+cardW, pts1[0][1]+cardH], [pts1[0][0], pts1[0][1]+cardH]])
    pts2 += offsetSize / 2
    M = cv2.getPerspectiveTransform(pts1,pts2) * 4

    
    transformed = np.zeros((int(cardW+offsetSize), int(cardH+offsetSize)), dtype=np.uint8);
    dst = cv2.warpPerspective(im, M, transformed.shape)
    return autocrop(dst)

def get_distance_line_pt(line0,p):
    x0,y0,x1,y1 = line0
    x2,y2 = p
    p0,p1,p2 = np.array([x0,y0]), np.array([x1,y1]), np.array([x2,y2])
    dist = np.cross(p1-p0,p2-p0)/np.linalg.norm(p1-p0)
    return dist
def check_closeness(line0,line1):
    x0,y0,x1,y1 = line0
    x2,y2,x3,y3 = line1
    p1 = np.array([x2,y2])
    p2 = np.array([x3,y3])
    p3 = (p1+p2)/2
    return min([get_distance_line_pt(line0,p) for p in (p1,p2,p3)])

def filter_lines(line0,lines,delta):
    delta = delta #* get_line_length(line0)
    for line in lines:
        if check_closeness(line0,line) < delta and angle(line)-angle(line0)<np.angle(20):
            lines.remove(line)
    return lines
def rcnn_rectify(im,predictor):
    #im = cv2.imread('dataset-cropped/Noah_21_01_01.jpg')
    outputs = predictor(im)
    result = outputs["instances"].to("cpu")
    img = result.pred_masks.numpy()[0].astype(np.uint8)
    mask = img * 255
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    hull = cv2.convexHull(cnt)
    epsilon = 0.005*cv2.arcLength(hull,True)
    approx = cv2.approxPolyDP(hull,epsilon,True)
    lines = get_lines(approx)

    lines_candidate = lines[:4]
    lines_candidate_2 = []
    try:
        for i in range(4):
            lines.sort(key=get_line_length,reverse=True)
            line0 = lines[0]
            lines_candidate_2.append(line0)
            lines.remove(line0)
            filter_lines(line0,lines,20)
        lines = lines_candidate_2
    except:
        lines = lines_candidate
    print(len(lines))
    print([angle(l) for l in lines])
 #   print(len(lines))
    # raise ValueError
    # [get_line_length(x) for x in lines]
    xb,yb,_ = im.shape
    try:
        key_points = get_intersectios(lines,xb,yb)
    except:
        return
    print(len(key_points))
    #raise ValueError
    if len(key_points)!=4:
        return
    key_points = orer_cordinates(key_points)

    rectified = transform_perspective(im,key_points)
    return rectified

class MaskRCNN:

    def __init__(self,model_path) -> None:
        self.predictor = load_predictor(model_path)

    def rectify(self,im):
        return rcnn_rectify(im,self.predictor)