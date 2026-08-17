# MinerU — Document Layout Detection (VLM)

MinerU (OpenDataLab) — это пайплайн извлечения содержимого документов. В этом
подходе используется **только его VLM-модель детекции макета**
`opendatalab/MinerU2.5-2509-1.2B` (Qwen2-VL fine-tune), которая на вход
принимает страницу и возвращает регионы с классами макета.

В отличие от DiT и LayoutLM-v3, MinerU **нет в unilm**, а его VLM обучалась на
собственной схеме DocStructBench / MinerU, **а не на PubLayNet**. Модель отвечает
своим словарём типов блоков, а **маппинг в начало `infer.py`
(`MINERU_BLOCK_TO_LABEL`)** сворачивает его в 5 классов PubLayNet:

| Класс MinerU | PubLayNet-класс | Русский |
|---|---|---|
| `text`, `caption`, `table_caption`, `image_caption`, `code_caption` | `text` | текст (включая подписи) |
| `title`, `doc_title`, `paragraph_title` | `title` | заголовки |
| `list`, `list_item` | `list` | списки |
| `table` | `table` | таблицы |
| `image`, `chart`, `image_block` | `figure` | картинки |

Идущие подряд блоки списка с близкой левой границей (одна ступень отступа, без
переноса) склеиваются в один регион `list` в `parse_layout_output()`.

Формат результата совпадает с DiT / LayoutLM-v3 (`pages` → `regions` →
`bbox`/`label`).

Репозиторий модели: <https://github.com/opendatalab/MinerU>

---

## Анализ железа и выбор способа запуска

| Параметр | Значение | Вывод |
|---|---|---|
| GPU | NVIDIA GeForce RTX 3050, 8 GB VRAM | Ampere, compute capability **8.6 (sm_86)** |
| Драйвер / CUDA | Driver 595.71.05, поддерживает **CUDA 13.2** | `nvcc` не нужен, PyTorch поставляет ядра |
| PyTorch (env venv) | **2.13.0+cu130** | CUDA runtime 13.0, совместим с драйвером |
| transformers (env) | **5.15.0** | есть `Qwen2VLForConditionalGeneration` / `AutoProcessor` |
| Диск | свободно **~177 GB** | хватает на веса (~2.3 GB) и зависимости |
| Python | 3.11.15 (pyenv), venv `env/` | используется готовый venv |

**Ключевой вывод по CUDA.** Версии согласованы «сверху вниз»: драйвер
(CUDA 13.2) ≥ runtime PyTorch (CUDA 13.0) ≥ архитектура GPU (sm_86). Компилятор
`nvcc` **не требуется**.

**Выбранный способ запуска — нативный (без Docker).** Причины те же, что у DiT:
данные и результаты работают по путям напрямую, в проекте уже есть venv `env/` с
`torch`, `transformers`, `pdf2image`, `pillow`, `numpy`, `huggingface_hub`.
Полный пакет `mineru[all]` (~20+ GB с paddleocr/vllm и др.) для одной лишь
детекции макета **не нужен** — используется только transformers-модель VLM.

Оценка ресурсов: MinerU2.5-2509-1.2B (~1.2B параметров, bf16) занимает
~2.3 GB на диске и ~2.5–4 GB VRAM при батче 1 (8 GB хватает с запасом). Одна
страница при изображении 1036×1036 обрабатывается за ~1–3 с на GPU.

---

## Установка (один раз)

```bash
# 1. Скачать предобученные веса VLM (~2.3 GB) локально
./approaches/MinerU/download_weights.sh
#   → approaches/MinerU/weights/{config.json, tokenizer*, model.safetensors, ...}

# 2. Требуемые пакеты уже стоят в env/: torch, transformers, pillow, numpy,
#    pdf2image, huggingface_hub, safetensors.
#    Проверка импорта:
env/bin/python -c "from transformers import Qwen2VLForConditionalGeneration, AutoProcessor; print('ok')"
```

> Веса качаются с HuggingFace `opendatalab/MinerU2.5-2509-1.2B` (зеркало на
> ModelScope — `OpenDataLab/MinerU2.5-2509-1.2B`). В отличие от DiT/LayoutLM-v3,
> здесь нужен полный снапшот (config + tokenizer + safetensors), а не один
> `.pth`, поэтому `download_weights.sh` использует `snapshot_download`.

---

## Запуск

```bash
# Формат: env/bin/python approaches/MinerU/infer.py <input_dir> <results_dir> [опции]
env/bin/python approaches/MinerU/infer.py \
  /home/daniil/pdfs_test/ \
  results_MinerU_publaynet/ \
  --device cuda --dpi 200
```

Опции:

| Аргумент | Назначение |
|---|---|
| `input_dir` | каталог с PDF и/или изображениями (`.jpg/.jpeg/.png`) |
| `results_dir` | каталог, куда пишутся JSON-результаты |
| `--model-dir` | локальный каталог со снапшотом VLM (пропускает скачивание) |
| `--device` | `cuda` (по умолчанию) или `cpu` |
| `--dpi` | DPI рендера PDF (по умолчанию `200`) |
| `--max-new-tokens` | максимум токенов генерации макета (по умолчанию `2048`) |

## Формат результата

Для каждого `foo.pdf`/`foo.jpg` создаётся `results_dir/foo.json`:

```json
{
  "pages": [
    {
      "num": 1,
      "regions": [
        {"bbox": [0.084, 0.111, 0.83, 0.715], "label": "table"},
        {"bbox": [0.085, 0.868, 0.4, 0.07], "label": "text"},
        {"bbox": [0.1, 0.05, 0.8, 0.03], "label": "title"}
      ],
      "metainfo": {"height": 792.0, "width": 612.0}
    }
  ]
}
```

- `bbox` — нормализованные `[x, y, w, h]` (x/y — левый верхний угол в долях
  ширины/высоты страницы).
- `metainfo.height/width` — размеры страницы в PDF-пойнтах (для изображений —
  пиксели).
- Классы PubLayNet: `text`, `title`, `list`, `table`, `figure`.

## Как устроена загрузка и инференс

1. `load_model()` — `Qwen2VLForConditionalGeneration.from_pretrained(weights_dir)`
   → `.to(device)` + `AutoProcessor.from_pretrained(weights_dir)`.
2. На каждой странице изображение приводится к RGB и ресайзится в
   `1036×1036` (размер, на котором MinerU делает layout detection).
3. Собираются chat-сообщения Qwen2-VL: системный промпт
   (`LAYOUT_SYSTEM_PROMPT`, вынесен в начало файла) + пользовательский триггер
   `Layout Detection:` (`LAYOUT_USER_PROMPT`, обученный триггер MinerU).
4. `model.generate(do_sample=False)` — жадная генерация в формате спец-токенов
   `<|box_start|>x0 y0 x1 y1<|box_end|><|ref_start|>type<|ref_end|>...`.
5. `parse_layout_output()` разбирает спец-токены, нормализует координаты
   0–1000 → `[x, y, w, h]` и сворачивает типы MinerU в PubLayNet через
   `MINERU_BLOCK_TO_LABEL` (подписи → `text`, `image`/`chart`/`image_block` →
   `figure`, `title`/`doc_title`/`paragraph_title` → `title`, `list`/`list_item`
   → `list`); текстовые блоки, полностью покрытые таблицей, отбрасываются
   (эвристика MinerU), а идущие подряд строки списка с близкой левой границей
   склеиваются в один регион `list`.

## Известные нюансы

- Модель — VLM, поэтому результат не детерминирован на 100% (зависит от версии
  transformers/токенизатора). Используется жадный декодинг (`do_sample=False`),
  чтобы минимизировать разброс.
- Модель отвечает своим словарём типов MinerU; типы вне маппинга (например
  `equation`/`code`/`header`/`footer`) молча отбрасываются, чтобы формат выхода
  совпадал с PubLayNet.
- На CPU модель работает (медленно); для GPU используется `bfloat16`.

## Ссылки

- Репозиторий: [opendatalab/MinerU](https://github.com/opendatalab/MinerU)
- Веса VLM: [opendatalab/MinerU2.5-2509-1.2B](https://huggingface.co/opendatalab/MinerU2.5-2509-1.2B)
- Статья MinerU: [arXiv:2409.18839](https://arxiv.org/abs/2409.18839)
