# Как запускать модели из microsoft/unilm — общий план

Этот файл — воспроизводимая инструкция для агентов, которые запускают модели
из репозитория [microsoft/unilm](https://github.com/microsoft/unilm)
(DiT, LayoutLM/LayoutLMv2/LayoutLMv3, LayoutXLM, TrOCR, BEiT и др.) на железе
этого проекта. План проверен на: `DiT/`.

---

## 0. Сначала — железо и CUDA (обязательно)

| Что проверить | Команда | На этой машине |
|---|---|---|
| GPU / VRAM | `nvidia-smi` | RTX 3050, 8 GB, sm_86 (Ampere) |
| Драйвер / CUDA | `nvidia-smi` (строка `CUDA Version`) | 595.71.05 → CUDA 13.2 |
| Версия torch | `env/bin/python -c "import torch; print(torch.__version__)"` | 2.13.0+cu130 |
| Python | `cat .python-version` | 3.11.15 (pyenv) |
| Диск | `df -h /` | ~14 GB свободно — критично |

**Правило совместимости CUDA (сверху вниз):**
`драйвер (13.2) ≥ runtime PyTorch (13.0) ≥ архитектура GPU (sm_86)`.

- `nvcc` / CUDA Toolkit **не нужен** — PyTorch уже содержит скомпилированные
  ядра под все архитектуры.
- `torch.cuda.is_available()` должен быть `True`; если нет — значит PyTorch
  собран под CPU, нужна переустановка `torch torchvision --index-url
  https://download.pytorch.org/whl/cuXXX`.

**Правило выбора способа запуска (натив vs Docker):**
- На этой машине свободно ~14 GB, поэтому **нативно** через готовый venv `env/`
  (образ `nvidia/cuda:*-devel` + torch + Detectron2 занимает 10–15 GB и не
  влезает). Docker использовать только если есть место и нужна изоляция.
- «Данные локально / результат локально» в нативном режиме работают по путям
  напрямую, без проброса томов.

---

## 1. Общая схема (для Detectron2-моделей)

Почти все doc-AI модели unilm для детекции/распознавания макета (DiT,
LayoutLMv3-detection, LayoutXLM) построены на **Detectron2** по одной схеме:

```
config.yaml + weights(.pth) + кастомные компоненты (ditod/…) + detectron2
```

План запуска любой такой модели:

1. Переиспользовать venv `env/`, проверить пакеты.
2. Найти/склонировать unilm (`approaches/unilm` — уже склонирован).
3. Найти в unilm: точку входа (`inference.py` / `train_net.py`), конфиг YAML,
   ссылку на веса.
4. Доставить недостающие зависимости (обычно `shapely`).
5. Применить Python 3.10+ фиксы (`collections.abc` и т.п.).
6. Скачать веса локально (Azure-ссылки из README мёртвые → зеркало на HF).
7. Написать `approaches/<Model>/infer.py` по эталону `DiT/infer.py`.
8. Запустить и проверить JSON-результат.

---

## 2. Окружение (переиспользуем, не пересоздаём)

Готовый venv уже лежит в `env/` и содержит почти всё:

```bash
env/bin/pip list | grep -iE "torch|detectron2|timm|transformers|numpy|pillow|pdf2image|huggingface"
```

Ожидаемо присутствуют: `torch 2.13.0+cu130`, `torchvision`, `detectron2 0.6`,
`timm`, `transformers`, `numpy`, `pillow`, `pdf2image`, `huggingface_hub`,
`fvcore`, `iopath`, `pycocotools`, `yacs`, `omegaconf`, `scipy`.

Чего **нет** и что обычно нужно доставить — `shapely` (тянет `ditod` через
модуль оценки таблиц):

```bash
env/bin/pip install shapely
```

> Проверка импорта пакета модели:
> ```bash
> env/bin/python -c "import sys; sys.path.insert(0, 'approaches/unilm/<model>/<subdir>'); import <model_pkg>"
> ```
> Если падает `ModuleNotFoundError` — доставить недостающий пакет и повторить.

---

## 3. Репозиторий unilm

Уже склонирован в `approaches/unilm`. Если нужен заново:

```bash
git clone https://github.com/microsoft/unilm.git approaches/unilm
```

Внутри структура — по одной папке на модель. Нужные подпапки обычно:
`<model>/inference.py`, `<model>/*_configs/*.yaml`, кастомный пакет с
регистрацией компонентов (например `dit/object_detection/ditod/`).

---

## 4. Веса: Azure-ссылки мёртвые → зеркало HuggingFace

Почти все ссылки на веса в README unilm ведут на
`layoutlm.blob.core.windows.net` и больше не отдаются публично
(HTTP 409). **Искать зеркало на HuggingFace:**

```bash
curl -s "https://huggingface.co/api/models?search=<model-name>" | python3 -c "import sys,json; [print(m['id']) for m in json.load(sys.stdin)]"
```

Известные рабочие зеркала:

| Модель | HF repo | Файл весов |
|---|---|---|
| DiT (детекция, PubLayNet) | `HYPJUDY/dit` | `dit-fts/publaynet_dit-b_mrcnn.pth` |
| DiT (pretrained backbone) | `HYPJUDY/dit` | `dit-pts/dit-base-224-p16-500k-62d53a.pth` |
| LayoutLMv3 (детекция, PubLayNet) | `HYPJUDY/layoutlmv3-base-finetuned-publaynet` | `model_final.pth` + `config.yaml` |
| DiT / BEiT (Transformers) | `microsoft/dit-base`, `microsoft/dit-base-finetuned-rvlcdip` | `pytorch_model.bin` |

Проверка размера перед скачиванием:

```bash
curl -sIL "https://huggingface.co/<repo>/resolve/main/<file>" | grep -i content-length
```

Скачивание в локальную папку подхода (папку `weights/` добавить в
`.gitignore`):

```bash
# через CLI
huggingface-cli download <repo> <file> --local-dir approaches/<Model>/weights
# или через Python API (fallback)
env/bin/python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='<repo>', filename='<file>', local_dir='approaches/<Model>/weights')"
```

---

## 5. Python 3.10+ фиксы в коде unilm

Старый код unilm написан под Python 3.7–3.9. Для 3.11 типичные правки:

```bash
# collections -> collections.abc
grep -rn "from collections import" approaches/unilm/<model>/ | grep -E "Iterable|Mapping|Sequence"
# заменить на: from collections.abc import Iterable  (и т.д.)
```

Проверенный пример — в `dit/object_detection/ditod/table_evaluation/
data_structure.py` фикс уже применён. Если при импорте появляются ошибки
`AttributeError`/`ImportError` про `collections` — это оно.

---

## 6. Пишем `infer.py` подхода (эталон — `DiT/infer.py`)

Для каждой новой модели создаём `approaches/<Model>/` и кладём туда:

- `infer.py` — скрипт инференса (единый интерфейс для всех подходов);
- `download_weights.sh` — скачивание весов;
- `README.md` — описание модели + инструкция;
- `weights/` — веса (в `.gitignore`).

**Единый CLI** (как в `DiT/infer.py` и `LayoutLM-v3/infer.py`):

```
env/bin/python approaches/<Model>/infer.py <input_dir> <results_dir> \
  [--model-dir DIR] [--device cuda|cpu] [--score-threshold 0.5] [--dpi 200]
```

**Структура `infer.py` (переиспользуемые куски из `DiT/infer.py`):**

1. `_find_unilm_root()` — поиск `approaches/unilm` по нескольким кандидатам.
2. `_register_*_components()` — добавить подпапку модели в `sys.path` и
   импортировать её пакет, чтобы зарегистрировать кастомные компоненты
   Detectron2 (backbone, ключи конфига).
3. `load_model()`:
   ```python
   cfg = get_cfg()
   add_*_config(cfg)              # кастомные ключи конфига модели
   cfg.merge_from_file(config_path)   # config_path — путь в unilm
   cfg.MODEL.WEIGHTS = str(weights_path)   # локальный путь весов
   cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
   cfg.MODEL.DEVICE = device
   predictor = DefaultPredictor(cfg)
   md = MetadataCatalog.get(cfg.DATASETS.TEST[0])
   md.set(thing_classes=[...])    # классы датасета (PubLayNet и т.п.)
   ```
4. `infer_pdf()` — `pdf2image.convert_from_path` → `np.array(img)` →
   `predictor(img)` → вытащить `instances.pred_boxes` и `pred_classes` →
   нормализовать bbox.
5. `main()` — glob `*.pdf` + `*.jpg/*.jpeg/*.png`, для каждого файла писать
   `results_dir/<name>.json`, в конце печатать `succeeded/failed`.

---

## 7. Запуск и проверка

```bash
# 1) smoke-тест на эталонном изображении из unilm (быстро, без PDF)
env/bin/python approaches/<Model>/infer.py \
  approaches/unilm/<model>/<example>.jpeg /tmp/<model>_smoke --device cuda

# 2) реальный запуск на PDF, результат — локально
env/bin/python approaches/<Model>/infer.py \
  /home/daniil/pdfs_test/ results_<Model>_publaynet/ --device cuda
```

Проверка результата:

```bash
python3 -c "import json; d=json.load(open('results_<Model>_publaynet/<file>.json')); \
  print(sum(len(p['regions']) for p in d['pages']), 'regions')"
```

Признаки успеха: процесс напечатал `Done. Processed N files (N succeeded,
0 failed)`, в JSON есть регионы с правдоподобными `bbox` и `label`.

---

## 8. Формат результата (единый для всех подходов)

```json
{
  "pages": [
    {
      "num": 1,
      "regions": [
        {"bbox": [0.1, 0.05, 0.8, 0.03], "label": "text"}
      ],
      "metainfo": {"height": 792.0, "width": 612.0}
    }
  ]
}
```

- `bbox` — нормализованные `[x, y, w, h]`, `x/y` — левый верхний угол в долях
  ширины/высоты страницы.
- `metainfo.height/width` — размеры страницы в PDF-пойнтах (для изображений —
  пиксели).
- Классы PubLayNet: `text`, `title`, `list`, `table`, `figure`.

---

## 9. Чек-лист типичных проблем

| Симптом | Причина | Решение |
|---|---|---|
| `ModuleNotFoundError: shapely` | `ditod` тянет оценку таблиц | `env/bin/pip install shapely` |
| `cannot import name 'Iterable' from 'collections'` | код под Py3.7–3.9 | `collections.abc` |
| `KeyError` / type mismatch при `merge_from_file` | кастомные ключи конфига не зарегистрированы | вызвать `add_*_config(cfg)` до `merge_from_file` |
| `HTTP 409 Public access is not permitted` | Azure-ссылка мёртвая | искать зеркало на HuggingFace (шаг 4) |
| `RuntimeError: CUDA out of memory` | батч/размер изображения велики | CPU или уменьшить `MIN_SIZE_TEST`, `--device cpu` |
| `cv2`/`opencv` отсутствует | оригинальный `inference.py` использует cv2 | в своём `infer.py` использовать `PIL`/`numpy`, а не cv2 |
| `from torch.utils.checkpoint` warnings | новое API torch | безвредно, игнорировать |

---

## Эталонные реализации

- `approaches/DiT/` — нативный запуск, Detectron2 + ditod (рабочий).
