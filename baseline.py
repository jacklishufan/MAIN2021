import argparse
import copy
import cv2 as cv
import glob
import json
import math
import ntpath
import numpy as np
import os
import sys
import warnings

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor

from utils.mask_detector import MaskRCNN
import utils.constants as constants
from utils.axis_detector import AxisDetector
from utils.gram_detector import GramDetector

from utils.rectification import rectify
# from rotation_correction import rotation_correction


def ransac(X, Y, residual_threshold=None):
    try:
        regressor = RANSACRegressor(residual_threshold=residual_threshold, max_trials=200).fit(X, Y)
    except ValueError as e:
        regressor = LinearRegression().fit(X, Y)

    return regressor


def show_fitted_line(slope, intercept, color, thickness):
    point1, point2 = (0, intercept), (image_to_draw.shape[1], slope * image_to_draw.shape[1] + intercept)
    point1, point2 = tuple(map(int, point1)), tuple(map(int, point2))
    _, point1, point2 = cv.clipLine((0, 0, image_to_draw.shape[1], image_to_draw.shape[0]), point1, point2)
    cv.line(image_to_draw, point1, point2, color, thickness)


def regression(image, bboxes, type_='frequency'):
    if type_ == 'frequency':
        x_coords = [(bbox[0] + bbox[2]) / 2 for bbox in bboxes]
        y_coords = [bbox[1] for bbox in bboxes]
    elif type_ == 'loss':
        x_coords = [bbox[2] for bbox in bboxes]  # It should be the right side of the bounding box for alignment reason
        y_coords = [(bbox[1] + bbox[3]) / 2 for bbox in bboxes]

    x_coords = [[x] for x in x_coords]
    y_coords = [[y] for y in y_coords]

    if type_ == 'frequency':
        residual_threshold = 0.01 * image.shape[0]
    elif type_ == 'loss':
        residual_threshold = 0.01 * image.shape[1]
        x_coords,y_coords = y_coords,x_coords

    return ransac(x_coords, y_coords, residual_threshold)


def check_gram(gram):
    if gram[0].shape[0] == 0 or gram[0].shape[0] > 2:
        return False

    return True


def check_axis(axis):
    frequency_num, loss_num = 0, 0
    for (_, cl) in zip(*axis):
        if 0 <= cl <= 7:
            frequency_num += 1
        elif cl >= 8:
            loss_num += 1

    if frequency_num < 2 or loss_num < 2:
        return False

    return True


def check_mark(mark):
    if mark[0].shape[0] == 0:
        return False

    return True


def spatial_to_value(slope, intercept, axis):
    bboxes = [bbox for (bbox, _) in axis]
    projected_coords = project(bboxes, slope, intercept)
    projected_coords = [x for (x, _) in projected_coords]
    v_coords = [cl for (_, cl) in axis]
    print("PROJECTED",projected_coords, v_coords)
    return projected_coords, v_coords


def project(bboxes, slope1, intercept, slope2=None):
    x_coords = [(bbox[0] + bbox[2]) / 2 for bbox in bboxes]
    y_coords = [(bbox[1] + bbox[3]) / 2 for bbox in bboxes]

    if slope1 == 0:
        return [(x, intercept) for x in x_coords]

    if slope2 is None:
        slope2 = 1. / slope1

    res = []
    for (x, y) in zip(x_coords, y_coords):
        b = y -slope2 * x - intercept
        res.append(((b / (slope1 - slope2)), (slope1 * (b / (slope1 - slope2)) + intercept)))

    return res


def category_to_text_label(categories, type_='frequency'):
    def custom_round(num):
        num = round(num * 2) / 2
        if num <= 2:
            num = round(num)

        return num

    res = []
    for category in categories:
        category = custom_round(category)
        if type_ == 'frequency':
            category = min(max(0, category), 7)
            if int(category) == category:
                res.append(int(125 * 2 ** category))
            else:
                res.append(int((125 * 2 ** math.floor(category) + 125 * 2 ** math.ceil(category)) / 2))
        elif type_ == 'loss':
            category = min(max(8, category), 21)
            if int(category) == category:
                res.append(int(-10 + 10 * (category - 8)))
            else:
                res.append(int((-10 + 10 * (math.floor(category) - 8) + (-10 + 10 * (math.ceil(category) - 8))) / 2))

    return res


def generate_result(mask_frequency, mask_loss, mask_category, res):
    res = copy.deepcopy(res)
    for (f, l, c) in zip(mask_frequency, mask_loss, mask_category):
        if c == 0:
            res['L'].append({'frequency': f, 'loss': l})
        elif c == 1:
            res['R'].append({'frequency': f, 'loss': l})

    return res


def baseline(axis_bbox_cl, mark_bbox_cl, image, res):
    res = copy.deepcopy(res)
    frequency_bbox_cl = sorted([(bbox, cl) for (bbox, cl) in axis_bbox_cl if 0 <= cl <= 7], key=lambda x: x[0][0])
    loss_bbox_cl = sorted([(bbox, cl) for (bbox, cl) in axis_bbox_cl if 8 <= cl <= 21], key=lambda x: x[0][0])
    frequency_bbox = [bbox for (bbox, _) in frequency_bbox_cl]
    loss_bbox = [bbox for (bbox, _) in loss_bbox_cl]
    mark_bbox = [bbox for (bbox, _) in mark_bbox_cl]
    mark_cl = [cl for (_, cl) in mark_bbox_cl]

    frequency_regressor = regression(image, frequency_bbox, type_='frequency')
    if isinstance(frequency_regressor, RANSACRegressor):
        slope_frequency, intercept_frequency, inlier_frequency_mask = frequency_regressor.estimator_.coef_[0][0], \
                                                                      frequency_regressor.estimator_.intercept_[0], \
                                                                      frequency_regressor.inlier_mask_
    elif isinstance(frequency_regressor, LinearRegression):
        slope_frequency, intercept_frequency, inlier_frequency_mask = frequency_regressor.coef_[0][0], \
                                                                      frequency_regressor.intercept_[0], \
                                                                      [True] * len(frequency_bbox)
    loss_regressor = regression(image, loss_bbox, type_='loss')
    if isinstance(loss_regressor, RANSACRegressor):
        slope_loss, intercept_loss, inlier_loss_mask = loss_regressor.estimator_.coef_[0][0],\
                                                       loss_regressor.estimator_.intercept_[0],\
                                                       loss_regressor.inlier_mask_
        w,b = slope_loss, intercept_loss
        # x = w * y + b=>y = 1/w * x - b / w
        slope_loss, intercept_loss = 1/w, -b/w
    elif isinstance(loss_regressor, LinearRegression):
        slope_loss, intercept_loss, inlier_loss_mask = loss_regressor.coef_[0][0], loss_regressor.intercept_[0], \
                                                       [True] * len(loss_bbox)

    if slope_frequency == slope_loss:
        warnings.warn('Axis detection fails')
        return res

    show_fitted_line(slope_frequency, intercept_frequency, (0, 255, 0), 2)
    show_fitted_line(slope_loss, intercept_loss, (0, 255, 0), 2)

    valid_frequency_bbox_cl = [t for i, t in enumerate(frequency_bbox_cl) if inlier_frequency_mask[i]]
    valid_loss_bbox_cl = [t for i, t in enumerate(loss_bbox_cl) if inlier_loss_mask[i]]
    valid_x_coords_frequency_xv, valid_v_coords_frequency_xv = spatial_to_value(slope_frequency, intercept_frequency,
                                                                                valid_frequency_bbox_cl)
    valid_x_coords_loss_xv, valid_v_coords_loss_xv = spatial_to_value(slope_loss, intercept_loss, valid_loss_bbox_cl)
    valid_x_coords_frequency_xv = [[x] for x in valid_x_coords_frequency_xv]
    valid_x_coords_loss_xv = [[x] for x in valid_x_coords_loss_xv]
    frequency_regressor_xv = ransac(valid_x_coords_frequency_xv, valid_v_coords_frequency_xv, residual_threshold=0.5)
    loss_regressor_xv = ransac(valid_x_coords_loss_xv, valid_v_coords_loss_xv, residual_threshold=0.5)

    projected_mark_frequency = project(mark_bbox, slope_frequency, intercept_frequency, slope_loss)
    projected_mark_loss = project(mark_bbox, slope_loss, intercept_loss, slope_frequency)
    x_coords_projected_mark_frequency = [[x] for (x, _) in projected_mark_frequency]
    x_coords_projected_mark_loss = [[x] for (x, _) in projected_mark_loss]
    print(x_coords_projected_mark_frequency,"RR")
    frequency_result = frequency_regressor_xv.predict(x_coords_projected_mark_frequency)
    print(frequency_result,"RR")
    loss_result = loss_regressor_xv.predict(x_coords_projected_mark_loss)

    frequency_result = category_to_text_label(frequency_result, type_='frequency')
    loss_result = category_to_text_label(loss_result, type_='loss')
    
    for (f, l, bbox) in zip(frequency_result, loss_result, mark_bbox):
        print(f, l, bbox)
        text = '{}, {}'.format(f, l)
        # print(image_to_draw, text, (bbox[0], bbox[1]))
        print(type(image_to_draw))
        cv.putText(image_to_draw, text, (int(bbox[0]), int(bbox[1])), cv.FONT_HERSHEY_PLAIN, (w2-w1)/500, (255, 0, 0)) 

    frequency_result, loss_result, mark_cl = [f for (f, l, c) in sorted(zip(frequency_result, loss_result, mark_cl))], \
                                             [l for (f, l, c) in sorted(zip(frequency_result, loss_result, mark_cl))], \
                                             [c for (f, l, c) in sorted(zip(frequency_result, loss_result, mark_cl))]

    res = generate_result(frequency_result, loss_result, mark_cl, res)

    return res


def main():
    parser = argparse.ArgumentParser()
    # if '--train' in sys.argv:
    #     parser.add_argument('--input_img_dir', type=str, default='/home/youwei/audiogram/detectron2/stage2/images/train')
        
    parser.add_argument('--input_img_dir', type=str, default='./input_images/test')
    parser.add_argument('--output_json_pth', type=str, default='./json_result/result_baseline.json')
    parser.add_argument('--output_img_dir', type=str, default='./output_images')
    parser.add_argument('--axis_detector_model', type=str, default='./models/axis_detector/model_final.pth')
    parser.add_argument('--gram_detector_model', type=str, default='./models/gram_detector/model_final.pth')
    parser.add_argument('--mask_model', type=str, default='./models/mask_detector/model_final.pth')
    parser.add_argument('--cpu', dest='cpu', action='store_true')
    parser.add_argument('--rectification', type=str, default='none')
    parser.set_defaults(cpu=False)
    args = parser.parse_args()
    mask_rcnn = MaskRCNN(args.mask_model)

    gram_detector = GramDetector(args.gram_detector_model, cpu=args.cpu)
    axis_detector = AxisDetector(args.axis_detector_model, cpu=args.cpu)

    output_json_dir = ntpath.dirname(args.output_json_pth)
    Path(output_json_dir).mkdir(parents=True, exist_ok=True)
    if os.path.isfile(args.output_json_pth):
        os.remove(args.output_json_pth)
        #raise FileExistsError('Json File Exists')

    global image_to_draw
    final_result = {}
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    filenames = []
    if os.path.isfile(args.input_img_dir):
        filenames.append(args.input_img_dir)
    elif os.path.isdir(args.input_img_dir):
        for ext in extensions:
            filenames.extend(sorted(glob.glob(os.path.join(args.input_img_dir, ext))))
    mis_count = []
    for filename in filenames:
        img_name = ntpath.basename(filename)
        print(img_name)
        image = cv.imread(filename)
        recognition_result = {"L": [], "R": []}

        [g_boxes, g_classes] = gram_detector.inference(image)
        # g_boxes = np.asarray([[0, 0, image.shape[1], image.shape[0]]])
        # g_classes = np.asarray([-1])
        if not check_gram([g_boxes, g_classes]):
            warnings.warn('No audiogram or more than two audiograms in {}'.format(img_name))
            final_result[img_name] = {"L": [], "R": []}
            continue

        global w1, h1, w2, h2
        for i, (g_box, g_category) in enumerate(zip(g_boxes, g_classes)):
            w1, h1, w2, h2 = g_box
            cropped_image = image[int(h1): int(h2)+1, int(w1): int(w2)+1]
            # cropped_image = rotation_correction(cropped_image)
            if args.rectification == 'vp':
            #Attempt to rectify using vanishing point Detection
                rectified_img = rectify(cropped_image)
                #Double Check if this rectification make sense, 
                #print(rectified_img)
                #cv.imwrite('rectified.png',rectified_img)
                [g_boxes_rec, g_classes_rec] = gram_detector.inference(rectified_img)
                if check_gram([g_boxes_rec, g_classes_rec]): 
                    #In case of failure, drop rectification
                    cropped_image = rectified_img
                else:
                    #raise ValueError
                    warnings.warn('Rectification Fails')
            elif args.rectification == 'mask':
                rectified_img = mask_rcnn.rectify(cropped_image)
                if rectified_img is None:
                    pass
                else:
                    [g_boxes_rec, g_classes_rec] = gram_detector.inference(rectified_img)
                    if check_gram([g_boxes_rec, g_classes_rec]) or 1: 
                        cropped_image = rectified_img
                    else:
                        warnings.warn('Rectification Fails')
            # print('max:', np.amax(cropped_image))
            image_to_draw = np.ndarray.astype(cropped_image.copy(), np.uint8)
            [out_boxes, out_classes] = axis_detector.inference(cropped_image)
            a_boxes = []
            a_classes = []
            m_boxes = []
            m_classes = []
            for box, category in zip(out_boxes, out_classes):
                #print(box,category)
                if category>=22:
                    m_boxes.append(box)
                    m_classes.append(category-22)
                else:
                    a_boxes.append(box)
                    a_classes.append(category)
            m_boxes = np.array(m_boxes)
            # [a_boxes, a_classes] = axis_detector.inference(cropped_image)
            if not check_axis([a_boxes, a_classes]):
                warnings.warn('Axis detection Fails')
                final_result[img_name] = {"L": [], "R": []}
                continue
            o = m_classes
            # [m_boxes, m_classes] = mark_detector.inference(cropped_image)
            # if not (len(o)==len(m_classes)):
            #     mis_count.append((filename,len(o),len(m_classes)))
            if not check_mark([m_boxes, m_classes]):
                warnings.warn('No mark on the audiogram')
                final_result[img_name] = {"L": [], "R": []}
                continue
            #print(a_boxes, a_classes)
            #idx = {}
            # for box, category in zip(a_boxes, a_classes):
            #     box = box.astype(int)
            #     cid = idx.get(category,0)+1
            #     idx[category] = cid
            #     cropped_label = image_to_draw[ int(box[1]): int(box[3])+1,int(box[0]): int(box[2])+1]
            #     cropped_label_img_to_draw_name = '{}_visualization_label_{}_{}_{}.jpg'.format(img_name.split('.')[0],str(i+1).zfill(2),category,cid)
            #     cv.imwrite(os.path.join(args.output_img_dir,'labels', cropped_label_img_to_draw_name), cropped_label)
        
            for box, category in zip(a_boxes, a_classes):
               

                box = box.astype(int)
                cv.rectangle(image_to_draw,
                             (box[0], box[1]),
                             (box[2], box[3]),
                             constants.AXIS_BBOX_COLORS[category],
                             thickness=3)
                cv.putText(image_to_draw, str(category), (int(box[0]), int(box[1])), cv.FONT_HERSHEY_PLAIN, (w2-w1)/500, (255, 0, 0)) 
            for box, category in zip(m_boxes, m_classes):
                box = box.astype(int)
                cv.rectangle(image_to_draw,
                             (box[0], box[1]),
                             (box[2], box[3]),
                             constants.MARK_BBOX_COLORS[category],
                             thickness=3)

            axis = [(bbox, cl) for (bbox, cl) in zip(*[a_boxes, a_classes])]
            mark = [(bbox, cl) for (bbox, cl) in zip(*[m_boxes, m_classes])]

            recognition_result = baseline(axis, mark, cropped_image, recognition_result)

            if len(g_boxes) >= 2:
                img_to_draw_name = '{}_visualization_{}.jpg'.format(img_name.split('.')[0], str(i+1).zfill(2))
            else:
                img_to_draw_name = '{}_visualization.jpg'.format(img_name.split('.')[0])
            print("Save image to ",os.path.join(args.output_img_dir, img_to_draw_name))
            cv.imwrite(os.path.join(args.output_img_dir, img_to_draw_name), image_to_draw)

        print(recognition_result)
        final_result[img_name] = recognition_result

    with open(args.output_json_pth, 'w') as json_file:
        json.dump(final_result, json_file, indent=2)
    for r in mis_count:
        print(*r)
    print()

if __name__ == '__main__':
    main()
