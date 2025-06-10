
from ultralytics import YOLO
import pandas as pd
import glob
from PIL import Image 
from sentence_transformers import SentenceTransformer
import numpy as np
import json
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#Sentencetranformer library performs embedding of images in a vector format
model_clip = SentenceTransformer('clip-Vit-B-32')

filedata_images = pd.read_csv('images.csv')
fd_prod_data = pd.read_excel('product_data.xlsx') 

#Using the NLP lib NLTK to analyze the scale of each video's description
sia = SentimentIntensityAnalyzer()

model = YOLO('yolov8m.pt')
#It helps in detecting objects from images and we're using MEGA model under it.

vid_id = '2025-05-28_13-42-32_UTC'
class_names_list = model.names
#The class_names_list stores the labels of all possible objects which YOLO could detect
#Example: {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',.....}

results = model.predict(source = 'videos/' + str(vid_id) + '.mp4', show = True, tracker='bytetrack.yaml', save_crop=True, stream=True, save=True, save_txt=True)
#The operation that is performed by YOLO will be visually displayed to user through another window

#At the same time the detected objects will be cropped and will be saved in the parent directory (i.e \runs\detect )
frm_num = 1
im_num = 1
return_val = []

for r in results:
    a = r.boxes 
    for i in range(len(a.cls)):
        temp_dict = {}
        temp_dict['Class_name'] = class_names_list[int(a.cls[i])]
        temp_dict['Bound_box'] = a.xywh[i]
        temp_dict['Confidence_num'] = a.conf[i]
        temp_dict['Frame_num'] = frm_num
        return_val.append(temp_dict)
        #The temp_dict stores details like class_name, bound_box(x,y,w,h), conf_num, frame_num
        #return_val list stores multiple dictionaries stored under temp_dict
    frm_num += 1


final_out = {'video_id': vid_id, 'vibes': [], 'products': [] }
#The above is the initialization of final_out dictionary which at last will be stored in a JSON file

with open('videos/' + str(vid_id) + '.txt', encoding='utf-8') as f:
    data = f.read()
    comp = sia.polarity_scores(str(data))
    #The compound score for each vibes are determined based on the meaning it has.
    #After collecting the meaning for all vibes, I set up a compound score range for each vibes 
    if (comp['compound'] <= 0.150):
        final_out['vibes'].append('Coquette')
    elif(comp['compound'] > 0.150 and comp['compound'] <= 0.300):
        final_out['vibes'].append('Street Wear')
    elif(comp['compound'] > 0.300 and comp['compound'] <= 0.550):
        final_out['vibes'].append('Clean Girl')
        final_out['vibes'].append('Y2K')
    elif(comp['compound'] > 0.550 and comp['compound'] <= 0.750):
        final_out['vibes'].append('Cottagecore')
        final_out['vibes'].append('Boho')
    else:
        final_out['vibes'].append('Party Glam')

image_paths = glob.glob(pathname='runs./**/*.jpg', recursive=True)
#Accessing all the saved images under runs folder(even from the sub-folders)

for index, image_path in enumerate(image_paths):
    #Each image_path will be stored having its own index value starting from 0
    image_1 = Image.open(str(image_path))
    #Opening the image(saved image) using PIL library 

    embed_1 = model_clip.encode(image_1)
    #The image(saved image) is embedded into vector form with the help of sentence_tranformers library
    for i in range(11689):
        image_2 = filedata_images.loc[i, 'image_url']
        #Now the URL image is accessed and stored in image_2 
        embed_2 = model_clip.encode(image_2)
        #The url image is embedded into vector form and stored in embed_2

        #Each saved image is compared with the list of all URL images with the help of cosine similarity
        dot_product = np.dot(np.array(embed_1), np.array(embed_2))
        magnitude_1 = np.sqrt(np.sum(embed_1**2))
        magnitude_2 = np.sqrt(np.sum(embed_2**2))

        #Using cosine_similarity to determine the similarity between the two images
        cosine_similarity = dot_product / (magnitude_1 * magnitude_2)
        
        product_temp = {}

        print(cosine_similarity)

        if(cosine_similarity >= 0.75):
            #If the cosine similarity if found to be greater than or equal to 0.75 then then the following operation will be executed
            pr_data_index = fd_prod_data.index[(fd_prod_data['id'] == filedata_images.loc[i, 'id'])].tolist()
            #Finding the index value of the 'id' in the product_data file 
            #That particular index value can be used later to access other datas in the product_data.xlsx file like 'product_type', 'product_tags', 'id'
            if not pr_data_index: 
                product_temp['type'] = str(fd_prod_data.loc[int(pr_data_index[0]), 'product_type']).lower()
                color_data = fd_prod_data.loc[pr_data_index[0], 'product_tags']
                product_temp['color'] = color_data.split(',')[0][7:].lower()
                product_temp['match_type'] = 'similar' if cosine_similarity <= 0.90 else 'exact'
                product_temp['matched_product'] = 'prod_'+str(fd_prod_data.loc[pr_data_index[0], 'id'])
            product_temp['confidence'] = "{:.2f}".format(cosine_similarity)
            final_out['products'].append(product_temp)

#Finally, a JSON file is created with the filename 'final_js.json' 
#All the datas stored in the final_out variable is copied to the JSON file
with open('final_js.json', 'w') as file:
    read_py = json.dump(final_out, file)
