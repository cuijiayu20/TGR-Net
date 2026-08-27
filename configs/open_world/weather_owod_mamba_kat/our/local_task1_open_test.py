_base_ = './local_voc20_clean.py'

class_names = [
    'person', 'car', 'bus', 'bicycle', 'motorbike',
    'aeroplane', 'boat', 'train', 'bottle', 'chair',
    'bird', 'cat', 'cow', 'dog', 'horse',
    'diningtable', 'pottedplant', 'sheep', 'sofa', 'tvmonitor'
]

prev_intro_cls = 0
cur_intro_cls = 5

test_dataloader = dict(
    num_workers=0,
    persistent_workers=False,
    dataset=dict(
        data_root='/data/weather/WeatherOWOD',
        ann_file='annotations/coco/task1/test_clean_ow.json',
        data_prefix=dict(img='')))

test_evaluator = dict(
    cfg=dict(
        dataset_root='/data/weather/WeatherOWOD/eval_voc/task1_test_clean',
        file_name='test_clean.txt',
        prev_intro_cls=prev_intro_cls,
        cur_intro_cls=cur_intro_cls,
        unknown_id=20,
        class_names=class_names))

val_dataloader = test_dataloader
val_evaluator = test_evaluator
