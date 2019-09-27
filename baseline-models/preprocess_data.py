import os
import itertools
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import imageio
import pandas as pd
from tqdm import tqdm


import warnings
warnings.filterwarnings("ignore")

include_folders = ['data/original/National Zoo',
 'data/original/Cincinnati Zoo',
 'data/original/Columbus Zoo']

data_folders = include_folders
labels = []

def get_labels(foldername):
    labels = [f for f in os.listdir(foldername)]
    return labels


data = {}
data['filepath']=[]
data['image']=[]
data['imagename'] = []
data['label'] = []
data["extension"] = []

extension = set()
empty_files = set()

## Loop through each image folder and extract all images
for image_folder in data_folders:
    for (dirpath, dirnames, filenames) in os.walk(image_folder):
        if dirnames == ".DS_Store":
            continue
            
        for f in tqdm(filenames):
            
            if f == ".DS_Store":
                continue
            
            ## check image
            size = os.stat(os.path.join(dirpath,f)).st_size
            
            if size == 0:
                empty_files.add(os.path.join(dirpath,f))
                continue
                
            ## found two images - odd images - Hank_IMG_6061.JPG, Connie_IMG_3589
            label = os.path.basename(dirpath)
            imagename = f"{label}_{f}"
            
            
            
            if imagename in ['Connie_IMG_3589.JPG','Hank_IMG_6061.JPG']:
                print("Ignoring image ",imagename)
                continue
            
            filename, file_extension = os.path.splitext(f)
            
            ## check image files
            if file_extension==".JPG":
                img = cv2.imread(os.path.join(dirpath, f))
                if img is None:
                    continue
            
                if len(img.shape)!=3:
                    print(img.shape)
                    continue
                    
            data["extension"].append(file_extension)
            data['label'].append(label)
            data['image'].append(f)
            data['imagename'].append(imagename)
            data['filepath'].append(os.path.join(dirpath, f))

df = pd.DataFrame(data)
print(f" No of files found: {df.shape[0]}")


# ## Exploratory Data Analysis

# In[12]:

## values
print(df["extension"].value_counts())


# In[47]:


df["extension"].value_counts().plot(kind="bar")
plt.savefig("images/binary_classifier_imagetypes.png")

# In[44]:


plt.figure(figsize=(20,10))
cnts = list(df["label"].value_counts())
cnts = [(i/sum(cnts))*100 for i in cnts]

#colors = ['b', 'g', 'r', 'c', 'm', 'y']

plt.pie(cnts, labels=df["label"].value_counts().keys(),
autopct='%1.1f%%',
counterclock=False, shadow=True)
plt.savefig("images/binary_classifier_label_distribution.png")


## Resize images

dim = (227, 227)
def getFrame(vidcap,sec,path):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        print('Original Dimensions : ',image.shape)
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(path, resized)     # save frame as JPG file
    return hasFrames

def get_images(save_path, video_file_path,filename):
    frameRate = 0.5 # capture image in each 0.5 second
    image_write_path  = lambda count: os.path.join(save_path,"%s_frame%d.jpg" % (filename,count))
    count=1
    sec = 0
    vidcap = cv2.VideoCapture(video_file_path)
    success = getFrame(vidcap,sec,image_write_path(count))
    while success:
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        success = getFrame(vidcap,sec,image_write_path(count))
        
        
def resize_image(imgpath,savepath):
    image = cv2.imread(imgpath, cv2.IMREAD_UNCHANGED)
    
    # resize image
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(savepath, resized)
    return 

save_path = os.path.join("data/binary-classifier/")
def save_images(df):
    for idx, row in tqdm(df.iterrows()):
        extension = row["extension"]
        current_file_path = row["filepath"]
        label = row["label"]
        newimagename = row["imagename"]
        
        if extension==".MOV":
            ## get frame and resize and save
            get_images(save_path, current_file_path,newimagename)
        elif extension==".JPG":
            #print(current_file_path)
            resize_image(current_file_path, os.path.join(save_path,newimagename))  
        else:
            print(current_file_path)
            continue
save_images(df)

print(f" No of files copied into binary classifier folder: {len(os.listdir(save_path))}")

## build csv file - 
csv_data = {}
csv_data["path"] = []

for (dirpath, dirnames, filenames) in os.walk(save_path):
        if dirnames == ".DS_Store":
            continue
            
        for f in filenames:
            
            if f == ".DS_Store":
                continue
            
            ## check image
            size = os.stat(os.path.join(dirpath,f)).st_size
            
            if size == 0:
                empty_files.add(os.path.join(dirpath,f))
                continue
                
            csv_data['path'].append(f)

csv_df = pd.DataFrame(csv_data)
csv_df.to_csv(os.path.join(save_path,"binary_classifier_imagepaths.csv"),index=False)
print("binary class csv files saved!")
print("preprocess all done!")