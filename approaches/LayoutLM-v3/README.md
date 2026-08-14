# LayoutLM-v3 — Document Layout Detection (Object Detection)

LayoutLMv3 — мультимодальный Transformer от Microsoft для Document AI с
унифицированным маскированием текста и изображений. Здесь он используется в
связке **Cascade Mask R-CNN (Detectron2) + LayoutLMv3-base backbone**
(`VLGeneralizedRCNN`), дообученной на **PubLayNet**, для детекции регионов
документа: `text`, `title`, `list`, `table`, `figure`.

Репозиторий модели: <https://github.com/microsoft/unilm/tree/master/layoutlmv3>

В отличие от DiT (backbone из timm), backbone LayoutLMv3 реализован как
transformers-модель (`layoutlmft.models.layoutlmv3.LayoutLMv3Model`), поэтому для
запуска нужны два пакета из unilm: `layoutlmft` и `ditod` (обе находятся внутри
`unilm/layoutlmv3/`). Режим инференса — **image-only** (`MODEL.IMAGE_ONLY: true`),
OCR-текст на вход не требуется.

---

## Анализ железа и выбор способа запуска

| Параметр | Значение | Вывод |
|---|---|---|
| GPU | NVIDIA GeForce RTX 3050, 8 GB VRAM | Ampere, compute capability **8.6 (sm_86)** |
| Драйвер / CUDA | Driver 595.71.05, поддерживает **CUDA 13.2** | `nvcc` не установлен, для PyTorch он не нужен |
| PyTorch (env venv) | **2.13.0+cu130** | CUDA runtime 13.0, совместим с драйвером 13.2 |
| transformers (env) | **5.15.0** | критично: нужны фиксы старого кода unilm (см. ниже) |
| Диск | свободно **~13 GB** на `/` | критично: Docker-образ CUDA devel не влезает |
| Python | 3.11.15 (pyenv), venv `env/` | используется готовый venv |

**Ключевой вывод по CUDA.** Версии согласованы «сверху вниз»: драйвер
(CUDA 13.2) ≥ runtime PyTorch (CUDA 13.0) ≥ архитектура GPU (sm_86). Компилятор
`nvcc` **не требуется** — PyTorch уже поставляет скомпилированные ядра.

**Выбранный способ запуска — нативный (без Docker).** Причины те же, что у DiT:
на диске ~13 GB свободно, образ CUDA devel + torch не влезет; в проекте уже есть
venv `env/` с `torch`, `torchvision`, `detectron2 0.6`, `transformers`,
`huggingface_hub`, `shapely`; данные и результаты работают по путям напрямую.

Оценка ресурсов: LayoutLMv3-base (~133M параметров backbone) + Cascade Mask
R-CNN занимает ~1.5 GB в VRAM на батче 1; одна страница при `MIN_SIZE_TEST=800`
обрабатывается за ~1–2 с на GPU.

---

## Установка (один раз)

Используется готовый venv проекта `env/`:

```bash
# 1. Клонировать unilm (если ещё нет) — подход использует код из
#    layoutlmv3/layoutlmft и layoutlmv3/examples/object_detection (ditod)
git clone https://github.com/microsoft/unilm.git approaches/unilm

# 2. Скачать предобученные веса (~538 MB) локально
./approaches/LayoutLM-v3/download_weights.sh
#   → approaches/LayoutLM-v3/weights/{model_final.pth, config.json, config.yaml}
```

Требуемые пакеты уже установлены в `env/`: `torch`, `torchvision`,
`detectron2`, `transformers`, `timm`, `numpy`, `pillow`, `pdf2image`,
`huggingface_hub`, `shapely`.

> Веса качаются с HuggingFace `HYPJUDY/layoutlmv3-base-finetuned-publaynet`
> (зеркало чекпоинта), потому что оригинальные ссылки на
> `layoutlm.blob.core.windows.net` больше не отдаются публично (HTTP 409).
> В одном каталоге обязательно должны лежать **оба** файла: `model_final.pth`
> (детектор) и `config.json` (конфиг transformers для backbone — он читается
> через `AutoConfig.from_pretrained(cfg.MODEL.CONFIG_PATH)`).

### Фиксы совместимости (уже применены в `approaches/unilm/layoutlmv3/`)

Старый код unilm написан под Python 3.7 и transformers 4.x. На Python 3.11 +
transformers 5.15 применены правки:

1. `modeling_layoutlmv3.py`: убраны удалённые из transformers 5 импорты
   `find_pruneable_heads_and_indices` / `prune_linear_layer`; `prune_heads`
   заменён на `NotImplementedError` (для инференса не нужен).
2. `modeling_layoutlmv3.py`: `self.init_weights()` заменён на
   `self.apply(self._init_weights)` (в transformers 5 `init_weights` теперь
   дополнительно тирует веса и требует атрибут `all_tied_weights_keys`).
3. `modeling_layoutlmv3.py`: вызов
   `get_extended_attention_mask(attention_mask, None, device)` → без третьего
   аргумента (в transformers 5 третьим позиционным параметром стал `dtype`,
   а не `device`).
4. `modeling_layoutlmv3.py`: `self.get_head_mask(...)` заменён локальной логикой
   (метод удалён в transformers 5).
5. `tokenization_layoutlmv3_fast.py`: импорт `RobertaTokenizerFast` из
   `transformers.models.roberta` (модуль `tokenization_roberta_fast` удалён).
6. `layoutlmft/models/layoutlmv3/__init__.py`: регистрация `AutoConfig/AutoModel`
   получила `exist_ok=True` (в transformers 5 `layoutlmv3` уже зарегистрирован
   нативно).
7. `ditod/table_evaluation/data_structure.py`: `from collections.abc import
   Iterable` (фикс Python 3.10+).

---

## Запуск

```bash
# Формат: env/bin/python approaches/LayoutLM-v3/infer.py <input_dir> <results_dir> [опции]
env/bin/python approaches/LayoutLM-v3/infer.py \
  /home/daniil/pdfs_test/ \
  results_LayoutLM-v3_publaynet/ \
  --device cuda --score-threshold 0.5 --dpi 200
```

Опции:

| Аргумент | Назначение |
|---|---|
| `input_dir` | каталог с PDF и/или изображениями (`.jpg/.jpeg/.png`) |
| `results_dir` | каталог, куда пишутся JSON-результаты |
| `--config-file` | путь к конфигу Detectron2 (по умолчанию `cascade_layoutlmv3.yaml` из unilm) |
| `--model-dir` | локальный каталог с `model_final.pth` + `config.json` (пропускает скачивание) |
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
        {"bbox": [0.084, 0.111, 0.83, 0.715], "label": "table"},
        {"bbox": [0.085, 0.868, 0.4, 0.07], "label": "text"}
      ],
      "metainfo": {"height": 792.0, "width": 600.12}
    }
  ]
}
```

`bbox` — нормализованные `[x, y, w, h]` (x/y — левый верхний угол в долях
ширины/высоты страницы). `metainfo.height/width` — размеры страницы в
PDF-пойнтах (для изображений — пиксели). Классы PubLayNet:
`text`, `title`, `list`, `table`, `figure`.

## Как устроена загрузка модели

1. `_find_unilm_root()` — находит `approaches/unilm`.
2. `_register_components()` — добавляет в `sys.path` каталоги
   `unilm/layoutlmv3` (пакет `layoutlmft`) и
   `unilm/layoutlmv3/examples/object_detection` (пакет `ditod`) и импортирует
   `ditod`, регистрируя в Detectron2 мета-архитектуру `VLGeneralizedRCNN`,
   backbone `build_vit_fpn_backbone` и кастомные ключи конфига
   (`MODEL.VIT`, `MODEL.IMAGE_ONLY`, `MODEL.CONFIG_PATH`, `AUG` и др.).
3. `load_model()` — `get_cfg()` → `add_vit_config(cfg)` → `merge_from_file`
   (`cascade_layoutlmv3.yaml`), затем подставляет локальный путь весов
   (`MODEL.WEIGHTS`), каталог с `config.json` (`MODEL.CONFIG_PATH`), устройство
   и порог. Модель строится через `DefaultPredictor`, чекпоинт грузится
   стандартным `DetectionCheckpointer` (веса `model_final.pth` уже в
   полном формате Detectron2: `backbone.bottom_up.backbone.*`,
   `backbone.fpn_*`, без префикса `layoutlmv3.`).

## Известные нюансы

- Backbone LayoutLMv3 — это transformers-модель, поэтому при запуске видны
  deprecation-предупреждения transformers (`use_return_dict`,
  `get_extended_attention_mask`) и `timm`/`torch.meshgrid` — они безвредны.
- Позиционные эмбеддинги ViT (`pos_embed`, 197 токенов под вход 224×224)
  интерполируются в `PatchEmbed.forward` под фактический размер изображения,
  поэтому инференс на страницах произвольного размера работает корректно.
- `MODEL.IMAGE_ONLY: true` — текст/OCR не используются, на вход подаётся только
  отрендеренная страница.
- Классы заданы жёстко (`text`, `title`, `list`, `table`, `figure`) и
  соответствуют `ROI_HEADS.NUM_CLASSES: 5` из конфига.

## Ссылки

- Статья: [LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking](https://arxiv.org/abs/2204.08387)
- Исходный репозиторий: [microsoft/unilm — layoutlmv3](https://github.com/microsoft/unilm/tree/master/layoutlmv3)
- Веса (зеркало): [HYPJUDY/layoutlmv3-base-finetuned-publaynet](https://huggingface.co/HYPJUDY/layoutlmv3-base-finetuned-publaynet)
