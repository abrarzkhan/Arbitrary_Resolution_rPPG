import os
import torch
import cv2
import numpy as np
import random
from torch.utils.data import Dataset
import json
from scipy import interpolate

class Normaliztion(object):
    def __call__(self, sample):
        video_x, video_y, clip_average_HR, ecg_label, frame_rate, scale = sample['video_x'], sample['video_y'], sample['clip_average_HR'], sample['ecg'], sample['frame_rate'], sample['scale']
        new_video_x = (video_x - 127.5) / 128
        new_video_y = (video_y - 127.5) / 128
        return {'video_x': new_video_x, 'video_y': new_video_y, 'clip_average_HR': clip_average_HR, 'ecg': ecg_label, 'frame_rate': frame_rate, 'scale':scale}

class RandomHorizontalFlip(object):
    def __call__(self, sample):
        video_x, video_y, clip_average_HR, ecg_label, frame_rate, scale = sample['video_x'], sample['video_y'], sample['clip_average_HR'], sample['ecg'], sample['frame_rate'], sample['scale']
        h, w = video_x.shape[1], video_x.shape[2]
        h1, w1 = video_y.shape[1], video_y.shape[2]
        new_video_x = np.zeros((video_x.shape[0], h, w, 3))
        new_video_y = np.zeros((video_y.shape[0], h1, w1, 3))
        p = random.random()
        if p < 0.5:
            for i in range(video_x.shape[0]):
                image = video_x[i, :, :, :]
                image = cv2.flip(image, 1)
                new_video_x[i, :, :, :] = image
                image1 = video_y[i, :, :, :]
                image1 = cv2.flip(image1, 1)
                new_video_y[i, :, :, :] = image1
            return {'video_x': new_video_x, 'video_y': new_video_y, 'clip_average_HR': clip_average_HR, 'ecg': ecg_label, 'frame_rate': frame_rate, 'scale':scale}
        else:
            return {'video_x': video_x, 'video_y': video_y, 'clip_average_HR': clip_average_HR, 'ecg': ecg_label, 'frame_rate': frame_rate, 'scale':scale}

class ToTensor(object):
    def __call__(self, sample):
        video_y = sample['video_y']
        video_x = sample['video_x']
        clip_average_HR = sample['clip_average_HR']
        ecg_label = sample['ecg']
        frame_rate = sample['frame_rate']
        scale= sample['scale']
        
        video_x = video_x.transpose((3, 0, 1, 2))
        video_x = np.array(video_x)
        video_y = video_y.transpose((3, 0, 1, 2))
        video_y = np.array(video_y)
        
        clip_average_HR = np.array(clip_average_HR)
        frame_rate = np.array(frame_rate)
        ecg_label = np.array(ecg_label)
        scale = np.array(scale)

        return {'video_x': torch.from_numpy(video_x.astype(np.float32)),
                'video_y': torch.from_numpy(video_y.astype(np.float32)),
                'clip_average_HR': torch.from_numpy(clip_average_HR.astype(np.float32)),
                'ecg': torch.from_numpy(ecg_label.astype(np.float32)),
                'frame_rate': torch.from_numpy(frame_rate.astype(np.float32)),
                'scale': torch.from_numpy(scale.astype(np.float32))}

class PURE_train(Dataset):
    def __init__(self, scale, frames, test=False, transform=None):
        self.train = not test
        self.path_json = "./custom_dataset/"
        self.path_data = "./custom_dataset/"
        self.length = frames
        self.scale = scale        
        self.idx_scale = 0
        self.idx_scale_vice = 0
        self.transform = transform
        
        # --- DYNAMIC TRAIN/TEST SPLIT (EXPANDED TO 105 CLIPS FOR CLIPS 95-104) ---
        all_clips = []
        for i in range(105):
            if os.path.exists(f"./custom_dataset/clip_{i:02d}"):
                all_clips.append(f"clip_{i:02d}")
            elif os.path.exists(f"./custom_dataset/clip_{i}"):
                all_clips.append(f"clip_{i}")
        
        if not test:
            # Keep original training behavior intact
            self.videoList = [clip for i, clip in enumerate(all_clips) if i % 5 != 0 and i < 95]
        else:
            # Test list automatically picks up validation splits AND clips 95-104
            self.videoList = [clip for i, clip in enumerate(all_clips) if (i % 5 == 0) or (i >= 95)]
        # -------------------------------------------------------------------------

        self.videoListSegNum = [] 
        self.videoListFrameRate = [24]  
        self.BVPList = [] 
        self.HRList = [] 
        
        for i in range(len(self.videoList)):
            tempPath = self.path_json + self.videoList[i] 
            picPath = tempPath + '/pic/1.0/'
            
            total_frames = len([name for name in os.listdir(picPath) if name.endswith(".png")])
            segNum = total_frames // self.length
            
            json_file = tempPath + '/' + self.videoList[i] + ".json"
            
            # Safe registration handling for newly recorded ungrounded clips
            if os.path.exists(json_file):
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    frame_data = data["/ImageData/FrameData"]
                    ppg = [item["Wave"] for item in frame_data]
            else:
                # Fallback dummies if you just want to run blind predictions on new test clips
                frame_data = [{"PulseRate": -1.0} for _ in range(total_frames)]
                ppg = [0.0 for _ in range(total_frames)]
            
            while len(ppg) < segNum * self.length:
                segNum -= 1

            self.videoListSegNum.append(segNum)   
            dataBvp = ppg[:segNum * self.length]
            self.BVPList.append([])
            self.HRList.append([])

            for j in range(segNum):
                startFrame = self.length * j
                segment_hr = frame_data[startFrame]["PulseRate"]
                self.BVPList[i].append(dataBvp[startFrame : startFrame + self.length])
                self.HRList[i].append([segment_hr])
       
        self.sampleCount = sum(self.videoListSegNum)
        self.sampleGetIdList = []
        temp = 0
        for num in self.videoListSegNum:
            temp += num
            self.sampleGetIdList.append(temp)    

    def __len__(self):
        return self.sampleCount

    def __getitem__(self, idx):
        for i in range(len(self.sampleGetIdList)):
            if(idx < self.sampleGetIdList[i]):
                vId = i
                clipId = idx - self.sampleGetIdList[i-1] if i != 0 else idx 
                break
                
        picPath = self.path_data + self.videoList[vId] + '/pic/'
        startFrame = clipId * self.length   
        
        video_x = self.get_single_video_x(picPath + "1.0/", startFrame)
        video_y = self.get_single_video_x(picPath + "1.0/", startFrame)

        frameRate = self.videoListFrameRate[0]
        ecgLabel = self.BVPList[vId][clipId]
        clipAverageHR = self.HRList[vId][clipId]

        sample = {'video_x': video_x, 'video_y': video_y, 'frame_rate': frameRate, 'ecg': ecgLabel, 'clip_average_HR': clipAverageHR, 'scale': self.scale[self.idx_scale]}

        if self.transform:
            sample = self.transform(sample)
        return sample

    def get_single_video_x(self, video_jpgs_path, start_frame):
        image_path = os.path.join(video_jpgs_path, '0.png')
        if not os.path.exists(image_path):
             image_path = os.path.join(video_jpgs_path, '1.png')
        
        image_shape = cv2.imread(image_path).shape
        video_x = np.zeros((self.length, image_shape[0], image_shape[1], 3))

        for i in range(self.length):
            s = start_frame + i
            image_name = str(s) + '.png'
            image_path = os.path.join(video_jpgs_path, image_name)

            tmp_image = cv2.imread(image_path)
            if tmp_image is None:
                tmp_image = np.zeros((image_shape[0], image_shape[1], 3))

            video_x[i, :, :, :] = tmp_image

        return video_x

    def set_scale(self, idx_scale, idx_scale_vice=0):
        self.idx_scale = idx_scale
        self.idx_scale_vice = idx_scale_vice