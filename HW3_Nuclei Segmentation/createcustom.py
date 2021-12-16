import glob

from convert2coco import *
from argparse import ArgumentParser 
# Label ids of the dataset
category_ids = {
    "nucleus": 1,
    
}

# Define which colors match which categories in the images
category_colors = {
    "(255, 255, 255)": 1 # nucleus,
}

# Define the ids that are a multiplolygon. In our case: wall, roof and sky
multipolygon_ids = [2, 5, 6]


def images_annotations_info_change(path):
    # This id will be automatically increased as we go
    annotation_id = 0
    image_id = 0
    annotations = []
    images = []
    aa=0
    for oriimage in os.listdir (path):
        original_file_name = oriimage
        original_file_name_forann = oriimage+'.png'
        tmp = os.path.join(path, original_file_name)
        mask_path = os.path.join(tmp, 'masks')

        image_path = os.path.join(path,original_file_name)
        tmp_ori = os.path.join(image_path,'images')
        tmp_ori = os.path.join(tmp_ori,original_file_name_forann)
        image_open = Image.open(tmp_ori).convert("RGB")
        widths, heights = image_open.size
        image = create_image_annotation(original_file_name_forann, widths, heights, image_id)
        images.append(image)
        for mask_image in glob.glob(os.path.join(mask_path , "*.png")):
            aa=aa+1
            # Open the image and (to be sure) we convert it to RGB
            mask_image_open = Image.open(mask_image).convert("RGB")
            w, h = mask_image_open.size
            
            # "images" info 
           

            sub_masks = create_sub_masks(mask_image_open, w, h)
            for color, sub_mask in sub_masks.items():
                category_id = category_colors.get(color)
                polygons, segmentations = create_sub_mask_annotation(sub_mask)

                for i in range(len(polygons)):
                    # Cleaner to recalculate this variable
                    segmentation = [np.array(polygons[i].exterior.coords).ravel().tolist()]
                    
                    annotation = create_annotation_format(polygons[i], segmentation, image_id, category_id, annotation_id)
                    
                    annotations.append(annotation)
                    annotation_id += 1
        image_id += 1
    return images, annotations, annotation_id

if __name__ == "__main__":
    # Get the standard COCO JSON format
    parser = ArgumentParser()
    parser.add_argument('--path', type=str)
    coco_format = get_coco_json_format()
    args=parser.parse_args()
    path = args.path

    d = os.listdir(path)
    
    coco_format["categories"] = create_category_annotation(category_ids)
    
    coco_format["images"], coco_format["annotations"], annotation_cnt = images_annotations_info_change(path)
   



    with open("output_train1.json","w") as outfile:
        json.dump(coco_format, outfile, indent=4)
    
    print("Created %d annotations for images in folder: %s" % (annotation_cnt, path))