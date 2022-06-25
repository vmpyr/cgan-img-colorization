from fastai.data.external import untar_data, URLs
coco_path = untar_data(URLs.COCO_SAMPLE)

print("The images of the dataset can be found here", coco_path)
print("Copy paste all of them to ./data/train")
