from detectron2.data.datasets import register_coco_instances
# for d in ["train", "test"]:
#     register_coco_instances(f"microcontroller_{d}", {}, f"Microcontroller Segmentation/{d}.json", f"Microcontroller Segmentation/{d}"

register_coco_instances('audiogram_gram_detection_train',{"thing_classes":['gram']},'annotations/gram_new_train.json','input_images/all')
register_coco_instances('audiogram_gram_detection_test',{"thing_classes":['gram']},'annotations/gram_new_test.json','input_images/all')


register_coco_instances('audiogram_axis_train',{},'annotations/gram_axis_mark_new_train.json','input_images/cropped')
register_coco_instances('audiogram_axis_test',{},'annotations/gram_axis_mark_new_test.json','input_images/cropped')

register_coco_instances('audiogram_segmentation_train',{"thing_classes":['chart']},'annotations/gram_mask_new_train.json','input_images/cropped')
register_coco_instances('audiogram_segmentation_test',{"thing_classes":['chart']},'annotations/gram_mask_new_test.json','input_images/cropped')

ORKA_DATASET = ['audiogram_gram_detection_train',
'audiogram_gram_detection_test',
'audiogram_axis_train',
'audiogram_axis_test',
'audiogram_segmentation_train',
'audiogram_segmentation_test']
