# Music Genre Classifier

Проект по классификации музыкальных треков по жанрам на основе mel spectrogram изображений.

Цель проекта: построить воспроизводимый ML pipeline, который покрывает подготовку данных, обучение модели, логирование экспериментов, версионирование данных и моделей, инференс на новых аудиофайлах, экспорт модели в ONNX и подготовку артефактов для production serving.

## Описание задачи

В проекте решается задача многоклассовой классификации музыкальных жанров. Исходные аудиофайлы переводятся в mel spectrogram представление, после чего задача сводится к классификации изображений спектрограмм.

Модель принимает на вход RGB изображение спектрограммы формы `[3, 224, 224]` и предсказывает один из 19 музыкальных жанров.

## Что реализовано

```text
1. Обучение модели на PyTorch Lightning
2. Конфигурация экспериментов через Hydra
3. Управление зависимостями через uv
4. Pre commit хуки для качества кода
5. Версионирование данных и моделей через DVC
6. Логирование метрик через MLflow
7. Сохранение checkpoint файлов обучения
8. Инференс по готовым spectrogram images
9. Инференс по реальному аудиофайлу
10. Экспорт двух обученных fold моделей в единую ONNX ensemble модель
11. Проверка ONNX модели через ONNX Runtime
12. Shell скрипт для конвертации ONNX в TensorRT
13. Triton model repository для ONNX Runtime backend
```

## Структура проекта

```text
music-genre-classifier/
├── configs/
│   ├── config.yaml
│   ├── data/
│   ├── dvc/
│   ├── export/
│   ├── inference/
│   ├── logging/
│   ├── model/
│   ├── preprocess/
│   ├── serving/
│   └── train/
├── music_genre_classifier/
│   ├── commands.py
│   ├── data.py
│   ├── export.py
│   ├── inference.py
│   ├── main.py
│   ├── model.py
│   └── train.py
├── plots/
│   ├── fold_0/
│   └── fold_1/
├── scripts/
│   └── export_tensorrt.sh
├── dataset.dvc
├── saved_model.dvc
├── exported_model.dvc
├── triton_model_repository.dvc
├── pyproject.toml
├── uv.lock
└── README.md
```

## Setup

### 1. Клонировать репозиторий

```bash
git clone https://github.com/IlyaYakovenko/music-genre-classifier.git
cd music-genre-classifier
```

### 2. Установить uv

Linux или macOS:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 3. Установить зависимости

```bash
uv sync
```

Проект использует `pyproject.toml` и `uv.lock`, поэтому окружение воспроизводится через lock файл.

### 4. Установить pre commit

```bash
uv run pre-commit install
uv run pre-commit run
```

Ожидаемый результат: хуки проходят успешно.

## DVC артефакты

В проекте используются DVC remotes для хранения больших артефактов отдельно от Git.

Основные DVC targets:

```text
dataset.dvc
saved_model.dvc
exported_model.dvc
triton_model_repository.dvc
```

Назначение targets:

```text
dataset.dvc хранит датасет
saved_model.dvc хранит обученные Lightning checkpoints
exported_model.dvc хранит ONNX ensemble модель
triton_model_repository.dvc хранит Triton model repository
```

В Git хранятся только `.dvc` метафайлы, а сами большие файлы подтягиваются через DVC.

### Скачать датасет

```bash
uv run dvc pull -r data_remote dataset.dvc
```

### Скачать обученные модели

```bash
uv run dvc pull -r models_remote saved_model.dvc
```

После этого появится структура:

```text
saved_model/
├── fold_0/
│   ├── config.yaml
│   └── checkpoints/
│       └── best.ckpt
└── fold_1/
    ├── config.yaml
    └── checkpoints/
        └── best.ckpt
```

### Скачать ONNX и Triton артефакты

```bash
uv run dvc pull -r models_remote exported_model.dvc
uv run dvc pull -r models_remote triton_model_repository.dvc
```

## Hydra конфигурация

Основная точка входа в конфиги:

```text
configs/config.yaml
```

Конфиги разделены по смысловым блокам:

```text
configs/data/
configs/model/
configs/train/
configs/logging/
configs/dvc/
configs/preprocess/
configs/inference/
configs/export/
configs/serving/
```

Посмотреть итоговый resolved config можно командой:

```bash
uv run music-genre-classifier show-config
```

## Train

Обучение запускается командой:

```bash
uv run music-genre-classifier train
```

Перед обучением команда может автоматически выполнить DVC pull датасета, если включено:

```yaml
dvc.pull_data_before_train: true
```

Пример короткого smoke test запуска:

```bash
uv run music-genre-classifier train data.max_items_per_class=2 train.batch_size=8 train.max_epochs=1 train.precision=32 dvc.pull_data_before_train=false
```

Пример полного запуска обучения:

```bash
uv run music-genre-classifier train train.max_epochs=10 train.batch_size=64 train.precision=16-mixed train.scheduler=onecycle
```

## Checkpoints

Во время обучения сохраняются:

```text
saved_model/fold_<id>/checkpoints/best.ckpt
saved_model/fold_<id>/checkpoints/last.ckpt
saved_model/fold_<id>/final.ckpt
```

Для итоговой версии проекта в DVC добавлены лучшие checkpoint файлы двух fold моделей:

```text
saved_model/fold_0/checkpoints/best.ckpt
saved_model/fold_1/checkpoints/best.ckpt
```

## Logging и результаты обучения

Для логирования используется MLflow.

Во время обучения логируются метрики:

```text
train_loss_step
train_loss_epoch
train_acc
train_f1_macro
val_loss
val_acc
val_f1_macro
lr-AdamW
```

Графики обучения сохранены в репозитории:

```text
plots/fold_0/
plots/fold_1/
```

Итоговые результаты двух обученных fold моделей:

```text
fold_0: val_acc около 0.358, val_f1_macro около 0.274
fold_1: val_acc около 0.371 to 0.378, val_f1_macro около 0.293 to 0.297
```

Модель не претендует на SOTA качество. Главный фокус проекта: воспроизводимый end to end MLOps pipeline.

## Infer

В проекте есть два режима инференса.

### Инференс по аудиофайлу

Команда принимает аудиофайл, строит mel spectrogram, прогоняет ensemble из двух fold моделей и сохраняет результат в JSON.

Если модели надо подтянуть из DVC:

```bash
uv run music-genre-classifier infer-audio inference.audio_path=sample_audio/example.wav inference.output_json=predictions/audio_prediction.json dvc.pull_model_before_infer=true
```

Если модели уже есть локально:

```bash
uv run music-genre-classifier infer-audio inference.audio_path=sample_audio/example.wav inference.output_json=predictions/audio_prediction.json dvc.pull_model_before_infer=false
```

Пример результата:

```json
{
  "filename": "example.wav",
  "predicted_class_id": 0,
  "predicted_genre": "Electronic",
  "confidence": 0.2244,
  "top_1_class_id": 0,
  "top_1_genre": "Electronic",
  "top_1_probability": 0.2244,
  "top_2_class_id": 3,
  "top_2_genre": "Experimental",
  "top_2_probability": 0.1921,
  "top_3_class_id": 10,
  "top_3_genre": "Ambient Electronic",
  "top_3_probability": 0.1888
}
```

### Инференс по готовым спектрограммам

Если уже есть PNG или JPG изображения спектрограмм:

```bash
uv run music-genre-classifier infer-images inference.input_dir=sample_dataset/test inference.output_csv=predictions/image_predictions.csv dvc.pull_model_before_infer=false
```

Результат сохраняется в CSV:

```text
predictions/image_predictions.csv
```

Файл содержит имя изображения, предсказанный класс, жанр, confidence и top k вероятности.

## ONNX export

Для production preparation два обученных fold checkpoint файла объединяются в единую ONNX ensemble модель.

Команда:

```bash
uv run music-genre-classifier export-onnx export.onnx_path=exported_model/music_genre_classifier_ensemble.onnx export.verify=true export.pull_model_before_export=false
```

ONNX модель принимает вход:

```text
input: [batch_size, 3, 224, 224]
```

и возвращает усреднённые вероятности двух fold моделей:

```text
probabilities: [batch_size, 19]
```

Экспортированная модель проверяется двумя способами:

```text
1. onnx.checker.check_model
2. сравнение выхода PyTorch ensemble и ONNX Runtime
```

В финальной проверке:

```text
ONNX verification max_abs_diff: 0.00000000
ONNX verification mean_abs_diff: 0.00000000
ONNX verification passed.
```

ONNX артефакты хранятся через DVC:

```text
exported_model/
├── music_genre_classifier_ensemble.onnx
├── music_genre_classifier_ensemble.onnx.data
└── music_genre_classifier_ensemble.metadata.json
```

Скачать их можно командой:

```bash
uv run dvc pull -r models_remote exported_model.dvc
```

## TensorRT

Для конвертации ONNX модели в TensorRT engine добавлен shell скрипт:

```text
scripts/export_tensorrt.sh
```

Он использует `trtexec` и конвертирует ONNX ensemble в serialized TensorRT engine:

```bash
bash scripts/export_tensorrt.sh exported_model/music_genre_classifier_ensemble.onnx exported_model/music_genre_classifier_ensemble.plan fp16
```

Внутри используется dynamic shape profile:

```text
min shape: input:1x3x224x224
opt shape: input:8x3x224x224
max shape: input:16x3x224x224
```

FP16 режим не требует calibration dataset. Поэтому дополнительные calibration data в DVC не добавлялись.

Важно: TensorRT `.plan` зависит от NVIDIA GPU, CUDA и версии TensorRT. Поэтому engine должен собираться в окружении с установленным TensorRT и доступным `trtexec`.

## Triton Inference Server

Для Triton подготовлен model repository:

```text
triton_model_repository/
└── music_genre_classifier/
    ├── config.pbtxt
    └── 1/
        └── model.onnx/
            ├── model.onnx
            └── music_genre_classifier_ensemble.onnx.data
```

Так как ONNX модель состоит из двух файлов, внутри Triton repository используется директория `model.onnx/`.

Конфигурация Triton:

```text
name: "music_genre_classifier"
platform: "onnxruntime_onnx"
max_batch_size: 16

input:
  name: "input"
  dtype: FP32
  dims: [3, 224, 224]

output:
  name: "probabilities"
  dtype: FP32
  dims: [19]

instance_group:
  KIND_CPU
```

ONNX внутри Triton repository был проверен локально:

```bash
uv run python -c "import onnx; p='triton_model_repository/music_genre_classifier/1/model.onnx/model.onnx'; m=onnx.load(p); onnx.checker.check_model(m); print('ONNX OK'); print([x.name for x in m.graph.input]); print([y.name for y in m.graph.output])"
```

Ожидаемый вывод:

```text
ONNX OK
['input']
['probabilities']
```

Triton repository хранится через DVC:

```bash
uv run dvc pull -r models_remote triton_model_repository.dvc
```

### Запуск Triton на CPU

Triton можно запустить через Docker image `nvcr.io/nvidia/tritonserver:24.12-py3`.

Для CPU запуска используется:

```text
model repository: triton_model_repository
model name: music_genre_classifier
backend: onnxruntime_onnx
instance group: KIND_CPU
HTTP port: 8000
GRPC port: 8001
metrics port: 8002
```

Проверка health endpoint:

```bash
curl -v http://localhost:8000/v2/health/ready
```

Проверка metadata:

```bash
curl -s http://localhost:8000/v2/models/music_genre_classifier/metadata
```

## Используемая модель

В проекте используется модель семейства XResNet и XSE ResNeXt для классификации изображений спектрограмм.

Особенности итогового решения:

```text
1. Задача сведена к image classification по mel spectrogram
2. Используется PyTorch Lightning модуль
3. Добавлены mixup, label smoothing и scheduler OneCycleLR
4. Реализованы DVC, Hydra, MLflow, checkpoints, ONNX export, TensorRT conversion script и Triton repository
5. Для инференса используется ensemble из двух обученных fold моделей
```

## Метрики

Основные метрики:

```text
validation accuracy
macro F1
cross entropy loss
```

Лучшее качество на двух fold моделях:

```text
fold_0: val_acc около 0.358, val_f1_macro около 0.274
fold_1: val_acc около 0.371 to 0.378, val_f1_macro около 0.293 to 0.297
```

Точность не является SOTA, но модель обучается, loss снижается, а весь end to end MLOps pipeline доведён до работающего состояния.

## Команды для проверки проекта

### Установка

```bash
uv sync
```

### Pre commit

```bash
uv run pre-commit install
uv run pre-commit run
```

### Скачать модели

```bash
uv run dvc pull -r models_remote saved_model.dvc
```

### Скачать ONNX

```bash
uv run dvc pull -r models_remote exported_model.dvc
```

### Скачать Triton repository

```bash
uv run dvc pull -r models_remote triton_model_repository.dvc
```

### Инференс на аудио

```bash
uv run music-genre-classifier infer-audio inference.audio_path=sample_audio/example.wav inference.output_json=predictions/audio_prediction.json dvc.pull_model_before_infer=false
```

### Инференс на готовых спектрограммах

```bash
uv run music-genre-classifier infer-images inference.input_dir=sample_dataset/test inference.output_csv=predictions/image_predictions.csv dvc.pull_model_before_infer=false
```

### ONNX export

```bash
uv run music-genre-classifier export-onnx export.onnx_path=exported_model/music_genre_classifier_ensemble.onnx export.verify=true export.pull_model_before_export=false
```

### TensorRT conversion

```bash
bash scripts/export_tensorrt.sh exported_model/music_genre_classifier_ensemble.onnx exported_model/music_genre_classifier_ensemble.plan fp16
```
