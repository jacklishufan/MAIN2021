import json
import numpy as np
import sys
import pandas as pd


def read_json(p):
    with open(p) as f:
        data = json.loads(f.read())
    return data

def count_acc(lst1,lst2):
        c_1 =dict(zip(*np.unique(lst1, return_counts=True)))
        c_2 =dict(zip(*np.unique(lst2, return_counts=True)))
        acc  = 0
        for k,v in c_1.items():
            acc += min(v,c_2.get(k,0))
        return acc

def compare_data(y_pred_path,y_truth_path):
    #y_pred_path = 'json_result/result_rectification_none.json'
    y_pred = read_json(y_pred_path)

   # y_truth_path = 'all_data_label.json'
    y_truth = read_json(y_truth_path)
    final_output = []
    organized_y_pred = {}
    for k,v in y_pred.items():
        for side in ['L','R']:
            if v[side]:
                lst = []
                for payload in v[side]:
                    freq = payload['frequency']
                    loss = payload['loss']
                    lst.append((freq,loss))
                organized_y_pred[(k,side)] = lst
                
    organized_y_truth = {}
    for k,v in y_truth.items():
        for side in ['L','R']:
            if v[side]:
                lst = []
                for payload in v[side]:
                    freq = payload[0]
                    loss = payload[1]
                    lst.append((freq,loss))
                organized_y_truth[(k,side)] = lst
    #ALL ACC
    ACC,TOL,REC = 0,0,0
    for k,v in organized_y_pred.items():
        if k not in organized_y_truth :
            print(k)
            continue
        acc = len([x for x in v if x in organized_y_truth[k]])
        tol = len(organized_y_truth[k])
        rec = len(v)
        ACC += acc
        TOL += tol
        REC += rec
    
    final_output.append(
        ("All Labels",ACC/TOL,ACC/REC)
    )
    # print("All Labels")
    # print(ACC,TOL,REC)
    # print(ACC/TOL)
    # print(ACC/REC)



    #FREQ
    ACC,TOL,REC = 0,0,0
    for k,v in organized_y_pred.items():
        if k not in organized_y_truth :
            print(k)
            continue
        freq_t = [x[0] for x in organized_y_truth[k]]
        freq_p = [x[0] for x in v]
        #acc = len([x for x in v if x[0] in freq_t])
        acc = count_acc(freq_p,freq_t)
        # try:
        #     assert len(set(freq_p))==len(freq_p)
        # except:
        #     print("W")
        #     print(freq_p)
        #     print("T")
        #     print(freq_t)
        tol = len(organized_y_truth[k])
        rec = len(v)
        ACC += acc
        TOL += tol
        REC += rec
    final_output.append(
        ("Frequency Labels",ACC/TOL,ACC/REC)
    )
    #Loss
    ACC,TOL,REC = 0,0,0
    for k,v in organized_y_pred.items():
        if k not in organized_y_truth :
            print(k)
            continue
        freq_t = [x[1] for x in organized_y_truth[k]]
        freq_p = [x[1] for x in v]
        #acc = len([x for x in v if x[0] in freq_t])
        acc = count_acc(freq_p,freq_t)
        # try:
        #     assert len(set(freq_p))==len(freq_p)
        # except:
        #     print("W")
        #     print(freq_p)
        #     print("T")
        #     print(freq_t)
        tol = len(organized_y_truth[k])
        rec = len(v)
        ACC += acc
        TOL += tol
        REC += rec

    final_output.append(
        ("Loss Labels",ACC/TOL,ACC/REC)
    )

    # +-5
    ACC,TOL,REC = 0,0,0
    TOTAL_ERR = []
    for k,v in organized_y_pred.items():
        if k not in organized_y_truth :
            print(k)
            continue
        freq_t = {x[0]:x[1] for x in organized_y_truth[k]}
        freq_p = {x[0]:x[1] for x in v}
        #acc = len([x for x in v if x[0] in freq_t])
        acc = len([1 for k,l in freq_p.items() if abs(freq_t.get(k,-1000)-l)<=5 ])
        t_err = [abs(freq_t.get(k,0)-l) for k,l in freq_p.items() if freq_t.get(k) is not None]
        TOTAL_ERR.extend(list(t_err))
        # try:
        #     assert len(set(freq_p))==len(freq_p)
        # except:
        #     print("W")
        #     print(freq_p)
        #     print("T")
        #     print(freq_t)
        tol = len(organized_y_truth[k])
        rec = len(v)
        ACC += acc
        TOL += tol
        REC += rec
    final_output.append(
        ("+-5 Loss Labels",ACC/TOL,ACC/REC)
    )
    return final_output


def main():
    ALL_PHOTO_PATH = 'annotations/camera_photo_freq_loss.json'
    SCANNED_IMG_PATH = 'annotations/scanned_image_freq_loss.json'
    true_pred_match = [
        [ALL_PHOTO_PATH,'json_result/result_baseline_rectification_none.json'],
        [ALL_PHOTO_PATH,'json_result/result_baseline_rectification_vp.json'],
        [ALL_PHOTO_PATH,'json_result/result_baseline_rectification_mask.json'],
        [SCANNED_IMG_PATH,'json_result/result_baseline_scanned.json'],
    ]
    for y_truth,y_pred in true_pred_match:
        res = compare_data(y_pred,y_truth)
        df = pd.DataFrame(res)
        df.columns = ['enrty','recall','precision']
        print(f"Summary of {y_pred}")
        print(df)

if __name__ == '__main__':
    main()