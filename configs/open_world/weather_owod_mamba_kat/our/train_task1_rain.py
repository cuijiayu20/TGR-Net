_base_ = './train_task1_fog.py'

work_dir = './work_dirs/weather_owod_mamba_kat_task1_rain'

train_dataloader = dict(
    dataset=dict(
        dataset=dict(ann_file='annotations/coco/task1/train_rain.json')))

test_dataloader = dict(
    dataset=dict(ann_file='annotations/coco/task1/test_rain_ow.json'))

test_evaluator = dict(
    cfg=dict(
        dataset_root='/data/weather/WeatherOWOD/eval_voc/task1_test_rain',
        file_name='test_rain.txt'))

val_dataloader = test_dataloader
val_evaluator = test_evaluator
