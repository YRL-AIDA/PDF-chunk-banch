# DiT — Document Layout Detection (Object Detection)

DiT (Document Image Transformer) — self-supervised предобученный ViT-подобный
backbone от Microsoft для задач Document AI. Здесь он используется в связке
**Mask R-CNN (Detectron2) + DiT-base backbone**, дообученной на **PubLayNet**,
для детекции регионов документа: `text`, `title`, `list`, `table`, `figure`.

Репозиторий модели: <https://github.com/microsoft/unilm/tree/master/dit/object_detection>

---

## Анализ железа и выбор способа запуска

| Параметр | Значение | Вывод |
|---|---|---|
| GPU | NVIDIA GeForce RTX 3050, 8 GB VRAM | Ampere, compute capability **8.6 (sm_86)** |
| Драйвер / CUDA | Driver 595.71.05, поддерживает **CUDA 13.2** | `nvcc` не установлен, но для PyTorch он не нужен |
| PyTorch (env venv) | **2.13.0+cu130** | CUDA runtime 13.0, полностью совместим с драйвером 13.2 |
| CPU / RAM | 12 потоков / 15 GB (свободно ~9 GB) | достаточно |
| Диск | свободно **~14 GB** на `/` | критично: Docker-образ CUDA devel не влезает |
| Python | 3.11.15 (pyenv), venv `env/` | используется готовый venv |

**Ключевой вывод по CUDA.** Версии согласованы «сверху вниз»: драйвер
(CUDA 13.2) ≥ runtime PyTorch (CUDA 13.0) ≥ архитектура GPU (sm_86).
Компилятор `nvcc` **не требуется** — PyTorch уже поставляет скомпилированные
ядра под все архитектуры. Отдельно устанавливать CUDA Toolkit не нужно.

**Выбранный способ запуска — нативный (без Docker).** Причины:

1. На диске всего ~14 GB свободно, а базовый образ `nvidia/cuda:12.4-devel`
   + `torch cu124` + Detectron2 (как в подходе LayoutLM-v3) занимает 10–15 GB
   и просто не влезет рядом с уже занятыми 856 GB.
2. В проекте уже есть готовый venv `env/` с установленными
   `torch 2.13.0+cu130`, `torchvision`, `detectron2 0.6`, `timm` — всё, что
   нужно DiT, уже стоит. Осталось добавить одну мелкую зависимость `shapely`.
3. «Данные локально / результат локально» в нативном режиме работают напрямую
   по путям, без проброса томов.

Оценка ресурсов для инференса: DiT-base + Mask R-CNN занимает ~0.9 GB в VRAM
на батче 1 (8 GB хватает с запасом), одна страница при `MIN_SIZE_TEST=800`
обрабатывается за ~0.5–2 с на GPU.

---

## Установка (один раз)

Используется готовый venv проекта `env/`:

```bash
# 1. Недостающая зависимость для импорта пакета ditod
env/bin/pip install shapely

# 2. Клонировать unilm (если ещё нет) — подход использует код из
#    dit/object_detection (ditod, конфиги)
git clone https://github.com/microsoft/unilm.git approaches/unilm

# 3. Скачать предобученные веса (~432 MB) локально
./approaches/DiT/download_weights.sh
#   → approaches/DiT/weights/dit-fts/publaynet_dit-b_mrcnn.pth
```

> Веса качаются с HuggingFace `HYPJUDY/dit` (зеркало чекпоинтов DiT), потому
> что оригинальные ссылки из README на `layoutlm.blob.core.windows.net` больше
> не отдают файлы публично (HTTP 409).

Требуемые пакеты уже установлены в `env/`: `torch`, `torchvision`,
`detectron2`, `timm`, `numpy`, `pillow`, `pdf2image`, `huggingface_hub`,
`shapely`.

---

## Запуск

```bash
# Формат: env/bin/python approaches/DiT/infer.py <input_dir> <results_dir> [опции]
env/bin/python approaches/DiT/infer.py \
  /home/daniil/pdfs_test/ \
  results_DiT_publaynet/ \
  --device cuda --score-threshold 0.5 --dpi 200
```

Опции (аналогично LayoutLM-v3):

| Аргумент | Назначение |
|---|---|
| `input_dir` | каталог с PDF и/или изображениями (`.jpg/.jpeg/.png`) |
| `results_dir` | каталог, куда пишутся JSON-результаты |
| `--config-file` | путь к конфигу Detectron2 (по умолчанию `maskrcnn_dit_base.yaml` из unilm) |
| `--model-dir` | локальный каталог с `publaynet_dit-b_mrcnn.pth` (пропускает скачивание) |
| `--device` | `cuda` (по умолчанию) или `cpu` |
| `--score-threshold` | порог уверенности (по умолчанию `0.5`) |
| `--dpi` | DPI рендера PDF (по умолчанию `200`) |

## Формат результата

Для каждого `foo.pdf`/`foo.jpg` создаётся `results_dir/foo.json`:

```json
{
  "pages": [
    {
      "num": 1,
      "regions": [
        {"bbox": [0.1, 0.05, 0.8, 0.03], "label": "text"},
        {"bbox": [0.2, 0.2, 0.5, 0.1], "label": "table"}
      ],
      "metainfo": {"height": 792.0, "width": 612.0}
    }
  ]
}
```

`bbox` — нормализованные `[x, y, w, h]` (x/y — левый верхний угол в долях
ширины/высоты страницы). Классы PubLayNet: `text`, `title`, `list`, `table`,
`figure`.

## Как устроена загрузка модели

1. `_find_unilm_root()` — находит `approaches/unilm`.
2. `_register_ditod_components()` — добавляет `dit/object_detection` в
   `sys.path` и импортирует `ditod`, регистрируя в Detectron2
   `build_vit_fpn_backbone` и кастомные ключи конфига (`MODEL.VIT`, `AUG` и др.).
3. `load_model()` — `get_cfg()` → `add_vit_config(cfg)` → `merge_from_file`
   (конфиг `maskrcnn_dit_base.yaml` с `_BASE_` → `Base-RCNN-FPN.yaml`),
   затем подставляет локальный путь весов, устройство и порог.

## Известные нюансы

- В `ditod/table_evaluation/data_structure.py` уже применён фикс
  `from collections import Iterable` → `from collections.abc import Iterable`
  (необходим для Python 3.10+).
- `import ditod` тянет модуль оценки таблиц, которому нужен `shapely` — он
  добавлен в шаг установки.
- Конфиг содержит `OUTPUT_DIR: $AMLT_OUTPUT_DIR` и `MODEL.WEIGHTS` со ссылкой на
  Azure — обе строки переопределяются/игнорируются в инференсе.

## Ссылки

- Статья: [DiT: Self-supervised Pre-training for Document Image Transformer](https://arxiv.org/abs/2203.02378)
- Исходный репозиторий: [microsoft/unilm](https://github.com/microsoft/unilm)
- Веса (зеркало): [HYPJUDY/dit](https://huggingface.co/HYPJUDY/dit)
