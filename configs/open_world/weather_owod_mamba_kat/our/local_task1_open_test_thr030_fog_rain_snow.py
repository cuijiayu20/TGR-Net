_base_ = './local_task1_open_test.py'

model = dict(test_cfg=dict(score_thr=0.30))

test_dataloader = dict(
    dataset=dict(ann_file='annotations/coco/task1/test_fog_rain_snow_ow.json'))

test_evaluator = dict(
    cfg=dict(
        dataset_root='/data/weather/WeatherOWOD/eval_voc/task1_test_fog_rain_snow',
        file_name='test_fog_rain_snow.txt'))

val_dataloader = test_dataloader
val_evaluator = test_evaluator
