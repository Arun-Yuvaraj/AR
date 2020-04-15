#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import cv2
import numpy as np
import subprocess
from collections import OrderedDict
from torch.autograd import Variable
import torch


# In[2]:


import json
from torch_utils import AverageMeter
import glob
import time
import argparse
from PIL import Image, ImageDraw, ImageFont


# In[3]:


from dataset import Video
from spatial_transforms import (Compose, Normalize, Scale, CenterCrop, ToTensor)
from temporal_transforms import LoopPadding


# In[4]:


from resnext import *

# In[5]:


ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', default='customer.mp4', help='path to the video file')
args = vars(ap.parse_args())


# In[6]:


use_cuda = torch.cuda.is_available()
sample_size = 112
sample_duration = 16
n_classes = 51
mean = [114.7748, 107.7354, 99.4750]
batch_size = 16 #32 if use_cuda else 16
n_threads = 4


# In[7]:


resnext_kinetics_hmdb_model_file = 'pretrained_models/resnext-101-kinetics-hmdb51_split1.pth'
resnext_kinetics_hmdb_model_file_64f = 'pretrained_models/resnext-101-64f-kinetics-hmdb51_split1.pth'

# In[5]:


device = torch.device('cuda:0' if use_cuda else 'cpu')
checkpoint = torch.load(resnext_kinetics_hmdb_model_file,map_location=device)
in_state_dict = checkpoint.pop('state_dict')


# In[6]:


out_state_dict = OrderedDict()


# In[7]:


for key, val in in_state_dict.items():
    s = '.'
    m = s.join((key.split(s))[1:])
    if m.find('downsample') < 0:
        out_state_dict[m] = val


# In[8]:


checkpoint['state_dict'] = out_state_dict


# In[9]:


model = resnet101(num_classes=n_classes, shortcut_type='A', sample_size=sample_size, sample_duration=sample_duration, last_fc=True)


# In[10]:


model.load_state_dict(out_state_dict)


# In[11]:


model.eval()


# In[169]:


spatial_transform = Compose([Scale(sample_size), CenterCrop(sample_size), ToTensor(), Normalize(mean, [1, 1, 1])])
temporal_transform = LoopPadding(sample_duration)


# In[ ]:


test_video = os.path.join('test_videos', args['video'])


# In[170]:


subprocess.call('mkdir tmp', shell=True)
subprocess.call('ffmpeg -i {} tmp/image_%05d.jpg'.format(test_video), shell=True)


# In[173]:


test_results = {'results': {}}
end_time = time.time()
output_buffer = []
previous_video_id = ''
batch_time = AverageMeter(name='Meter', length=10)
data_time = AverageMeter(name='Meter', length=10)


# In[171]:


data = Video('tmp', spatial_transform=spatial_transform, temporal_transform=temporal_transform,
                 sample_duration=sample_duration)


# In[172]:


data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


# In[174]:


videoPath = "../dataset/{}/*".format("hmdb") 
activity_classes = [i.split(os.path.sep)[3] for i in glob.glob(videoPath)]      
print(activity_classes)


# In[176]:

def calc_frames_per_sec(video_file_path, frames_directory_path):
    p = subprocess.Popen('ffprobe {}'.format(video_file_path), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _, res = p.communicate()
    res = res.decode('utf-8')

    duration_index = res.find('Duration:')
    duration_str = res[(duration_index + 10):(duration_index + 21)]
    hour = float(duration_str[0:2])
    minute = float(duration_str[3:5])
    sec = float(duration_str[6:10])
    total_sec = hour * 3600 + minute * 60 + sec

    n_frames = len(os.listdir(frames_directory_path))
    fps = round(n_frames / total_sec, 2)
    return fps


def calculate_video_results(output_buffer, video_id, test_results, activity_classes):
    video_outputs = torch.stack(output_buffer)
    average_scores = torch.mean(video_outputs, dim=0)
    sorted_scores, locs = torch.topk(average_scores, k=10)

    video_results = []
    for i in range(sorted_scores.size(0)):
        video_results.append({'label': activity_classes[locs[i]], 'score': sorted_scores[i].detach().numpy().tolist()})
    
    test_results['results'][np.array_str(video_id.detach().numpy())] = video_results


# In[181]:



detected_actions = {}
unit_classes = []
unit_segments = []
for i, (inputs, targets) in enumerate(data_loader):
    data_time.update(time.time() - end_time)
    inputs = Variable(inputs, volatile=True)
    outputs = model(inputs)
        
    for j in range(outputs.size(0)):
        if not (i == 0 and j == 0):
            calculate_video_results(output_buffer, previous_video_id, test_results, activity_classes)
            print("------------------------------------------------------------")
            print(np.array_str(previous_video_id.detach().numpy()))
            top_preds = test_results['results'][np.array_str(previous_video_id.detach().numpy())]
            print(top_preds)
            output_buffer = []
            detected_actions[top_preds[0]['label']] = detected_actions.get(top_preds[0]['label'],0) + 1
            # unit_classes.append("Top 2 Actions: " + top_preds[0]['label'] + " / " + top_preds[1]['label'])
            unit_classes.append(top_preds[0]['label'])
            unit_segments.append(previous_video_id.detach().numpy())
        output_buffer.append(outputs[j].data.cpu())
        previous_video_id = targets[j]

    if (i % 100) == 0:
        with open(os.path.join('pred_results','output.json'),'w') as f:
            json.dump(test_results, f)    

    batch_time.update(time.time() - end_time)
    end_time = time.time()

    print('[{}/{}]\t Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                  i + 1, len(data_loader), batch_time=batch_time, data_time=data_time))
    with open(os.path.join('pred_results','output.json'),'w') as f:
        json.dump(test_results, f)


fps = calc_frames_per_sec(test_video, 'tmp')


for i in range(len(unit_classes)):
    for j in range(unit_segments[i][0], unit_segments[i][1] + 1):
        image = Image.open('tmp/image_{:05}.jpg'.format(j)).convert('RGB')
        min_length = min(image.size)
        font_size = int(min_length * 0.05)
        font = ImageFont.truetype(os.path.join(os.path.dirname(__file__),'SourceSansPro-Regular.ttf'),font_size)
        d = ImageDraw.Draw(image)
        textsize = d.textsize(unit_classes[i], font=font)
        x = int(font_size * 0.5)
        y = int(font_size * 0.25)
        x_offset = x
        y_offset = y
        rect_position = (x, y, x + textsize[0] + x_offset * 2, y + textsize[1] + y_offset * 2)
        d.rectangle(rect_position, fill=(30, 30, 30))
        d.text((x + x_offset, y + y_offset), unit_classes[i],font=font, fill=(235, 235, 235))

            
        image.save('tmp/image_{:05}_pred.jpg'.format(j))

dst_file_path = os.path.join('pred_results', args['video'])
subprocess.call('ffmpeg -y -r {} -i tmp/image_%05d_pred.jpg -b:v 1000k {}'.format(fps, dst_file_path), shell=True)



# In[178]:


print("List of detected actions in the video:")
print(detected_actions)


# In[191]:


subprocess.call('rm -rf tmp', shell=True)


# In[16]:
