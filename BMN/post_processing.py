# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import json
import multiprocessing as mp

from utils import iou_with_anchors


def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data


def getDatasetDict(opt):
    df = pd.read_csv(opt["video_info"])
    json_data = load_json(opt["video_anno"])
    database = json_data
    video_dict = {}
    for i in range(len(df)):
        video_name = df.video.values[i]
        video_info = database[video_name]
        video_new_info = {}
        video_new_info['duration_frame'] = video_info['duration_frame']#根据csv的name获取json中的信息
        video_new_info['duration_second'] = video_info['duration_second']
        video_new_info["feature_frame"] = video_info['feature_frame']
        video_subset = df.subset.values[i]                                                 #根据csv判断是train还是验证
        video_new_info['annotations'] = video_info['annotations']         #segment + label
        #if video_subset == 'validation':
        if video_subset == 'test': #                         8.11
            video_dict[video_name] = video_new_info
    return video_dict        #看一下，是不是取了validation的所有信息


def soft_nms(df, alpha, t1, t2):
    '''
    df: proposals generated by network;
    alpha: alpha value of Gaussian decaying function;
    t1, t2: threshold for soft nms.
    '''
    df = df.sort_values(by="score", ascending=False)
    tstart = list(df.xmin.values[:])
    tend = list(df.xmax.values[:])
    tscore = list(df.score.values[:])

    rstart = []
    rend = []
    rscore = []

    #while len(tscore) > 1 and len(rscore) < 101:
    while len(tscore) > 1 and len(rscore) < 41: #                      1111111111111111111111111111
        max_index = tscore.index(max(tscore))
        #print('max_index')#                                                       11111111111111
        #print(max_index)
        tmp_iou_list = iou_with_anchors(
            np.array(tstart),
            np.array(tend), tstart[max_index], tend[max_index])
        for idx in range(0, len(tscore)):
            if idx != max_index:
                tmp_iou = tmp_iou_list[idx]
                tmp_width = tend[max_index] - tstart[max_index]
                if tmp_iou > t1 + (t2 - t1) * tmp_width:
                    tscore[idx] = tscore[idx] * np.exp(-np.square(tmp_iou) /
                                                       alpha)

        rstart.append(tstart[max_index])
        rend.append(tend[max_index])
        rscore.append(tscore[max_index])
        tstart.pop(max_index)
        tend.pop(max_index)
        tscore.pop(max_index)

    newDf = pd.DataFrame()
    newDf['score'] = rscore
    newDf['xmin'] = rstart
    newDf['xmax'] = rend
    return newDf


def video_post_process(opt, video_list, video_dict):
    for video_name in video_list:
        #df = pd.read_csv("./output/BMN_results/" + video_name + ".csv")
        df = pd.read_csv("./output/BMN_results1/" + video_name + ".csv") #          8.1111111111

        if len(df) > 1:
            snms_alpha = opt["soft_nms_alpha"]
            snms_t1 = opt["soft_nms_low_thres"]
            snms_t2 = opt["soft_nms_high_thres"]
            df = soft_nms(df, snms_alpha, snms_t1, snms_t2)

        df = df.sort_values(by="score", ascending=False)
        video_info = video_dict[video_name]
        #video_duration = float(video_info["duration_frame"] // 16 * 16) / video_info["duration_frame"] * video_info["duration_second"]
        video_duration = float(video_info["duration_frame"] ) / video_info["duration_frame"] * video_info["duration_second"]
        proposal_list = []
        
        #print('len(df)') #111111111111111111111111111111111111111
        #print(len(df))
        #for j in range(min(100, len(df))):
        #for j in range(min(2, len(df))):#                                    111111111111111111111111
        for j in range(min(1, len(df))):
            tmp_proposal = {}
            tmp_proposal["score"] = df.score.values[j]
            tmp_proposal["segment"] = [max(0, df.xmin.values[j]) * video_duration,
                                       min(1, df.xmax.values[j]) * video_duration]
            proposal_list.append(tmp_proposal)
        #result_dict[video_name[2:]] = proposal_list
        result_dict[video_name[0:]] = proposal_list


def BMN_post_processing(opt):
    video_dict = getDatasetDict(opt)
    video_list = list(video_dict.keys())  # [:100]
    #print('len(video_list)')   # 长度是55
    #print(len(video_list))#11111111111111111111111111111111111
    global result_dict
    result_dict = mp.Manager().dict()

    num_videos = len(video_list)
    num_videos_per_thread = num_videos // opt["post_process_thread"]    #opt里的值8
    processes = []
    for tid in range(opt["post_process_thread"] - 1):
        tmp_video_list = video_list[tid * num_videos_per_thread:(tid + 1) * num_videos_per_thread]
        p = mp.Process(target=video_post_process, args=(opt, tmp_video_list, video_dict))
        p.start()
        processes.append(p)
    tmp_video_list = video_list[(opt["post_process_thread"] - 1) * num_videos_per_thread:]
    p = mp.Process(target=video_post_process, args=(opt, tmp_video_list, video_dict))
    p.start()
    processes.append(p)
    for p in processes:
        p.join()

    result_dict = dict(result_dict)
    output_dict = {"version": "VERSION 1.3", "results": result_dict, "external_data": {}}
    outfile = open(opt["result_file"], "w")
    json.dump(output_dict, outfile)
    outfile.close()

# opt = opts.parse_opt()
# opt = vars(opt)
# BSN_post_processing(opt)