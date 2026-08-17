# DeepSeek-OCR — Document Layout Detection (prompt-based grounding)

DeepSeek-OCR (`deepseek-ai/DeepSeek-OCR`) — VLM для оптического сжатия документов
(«Contexts Optical Compression»). Здесь он используется **только для детекции
макета**: модель просят вернуть координаты регионов документа.

Главное отличие от остальных подходов проекта:

| Подход | Как получаются классы |
|---|---|
| DiT / LayoutLM-v3 | детектор Detectron2 предсказывает 5 классов PubLayNet напрямую |
| MinerU | VLM отвечает *своим* словарём блоков, который **маппится** в PubLayNet |
| **DeepSeek-OCR** | классов в модели нет вообще — их **запрашивают промптом** (`LAYOUT_PROMPT`) |

Промпт вынесен **в самое начало** `infer.py` (константа `LAYOUT_PROMPT`) и в нём
подробно описано, что считать таблицей, текстом, заголовком, списком (**список
целиком, а не отдельный айтем**) и рисунком (**фрагменты рисунка объединяются в
одну фигуру**). Модель отвечает в своём grounding-формате:

```
<|ref|>CLASS<|/ref|><|det|>[[x1, y1, x2, y2]]<|/det|>
```

с координатами в диапазоне 0–999. Классы называются каноническими именами
PubLayNet: `text`, `title`, `list`, `table`, `figure`.

Репозиторий модели: <https://github.com/deepseek-ai/DeepSeek-OCR>

---

## Анализ железа и выбор способа запуска

| Параметр | Значение | Вывод |
|---|---|---|
| GPU | NVIDIA GeForce RTX 3050, **8 GB VRAM** | Ampere, compute capability **8.6 (sm_86)** |
| Драйвер / CUDA | Driver 595.71.05, поддерживает **CUDA 13.2** | `nvcc` не нужен |
| PyTorch (env venv) | **2.13.0+cu130** | CUDA runtime 13.0, совместим с драйвером |
| transformers (env) | **5.15.0** | у DeepSeek-OCR требование **4.46.3** — см. ниже |
| Диск | свободно **~175 GB** | хватает на веса (~6.7 GB) |
| Python | 3.11.15 (pyenv), venv `env/` | используется готовый venv |

**Ключевой вывод по CUDA.** Драйвер (CUDA 13.2) ≥ runtime PyTorch (CUDA 13.0) ≥
архитектура GPU (sm_86) — порядок согласован, `nvcc`/CUDA Toolkit не нужны.

**Ключевой вывод по VRAM — модель на грани для 8 GB.** DeepSeek-OCR — это MoE
LLM-бэкбон DeepSeek-V2 (hidden 1280, 12 слоёв, 64 routed experts + 2 shared) +
энкодеры SAM-ViT-B и CLIP-L. Полный снапшот в bf16 весит **~6.7 GB**; плюс
фичи-мапы энкодеров, активации и пик logits `[1, seq, 129280]` в fp32 при
генерации. Официально модель демонстрируется на **A100-40G**.

**Проверено на этой машине (RTX 3050, 8 GB):**
- bf16-веса на GPU занимают **6.77 GB** (после исправления порядка загрузки, см.
  ниже), свободно остаётся ~0.79 GB.
- `crop_mode=False, base_size=640` (default) — **работает** (100 vision-токенов).
- `crop_mode=True, base_size=1024` (полное Gundam) — **OOM** (856 vision-токенов,
  пик logits ~610 MiB). Нужен GPU ≥ 12–16 GB или `--quantize 8bit`.

> **Исправление загрузки (критично).** Раньше `load_model()` делал
> `.to(device).to(dtype)`: `from_pretrained` без `torch_dtype` грузил fp32
> (~13.3 GB), и `.to("cuda")` падал с `CUDA out of memory`. Теперь модель грузится
> сразу в `torch_dtype=torch.bfloat16` и переносится на GPU **после** каста —
> на GPU уходит уже bf16 (~6.7 GB).

**Способ запуска — нативный (без Docker).** Причины те же, что у DiT/MinerU:
данные и результаты работают по путям напрямую. Модель грузится через
`transformers` + `trust_remote_code`, полный образ CUDA/vLLM не нужен.

> **Версия transformers — обязателен отдельный venv.** Remote-код DeepSeek-OCR
> написан под `transformers==4.46.3` (см. `requirements.txt` репозитория), а в
> venv проекта стоит `5.15.0`. Проверено: на 5.15.0 импорт remote-кода падает с
>
> ```
> ImportError: cannot import name 'LlamaFlashAttention2' from 'transformers.models.llama.modeling_llama'
> ```
>
> (в transformers 5.x класс `LlamaFlashAttention2` переименован). Поэтому
> заводим отдельное окружение:
>
> ```bash
> python3 -m venv env_dsk && source env_dsk/bin/activate
> pip install -r approaches/DeepSeek-OCR/requirements.txt
> ```
>
> (в `requirements.txt` уже включён `torch==2.6.0` — PyPI-сборка на Linux x86_64
> это CUDA 12.4; `flash-attn` не обязателен — в `infer.py` он **не**
> используется, чтобы не тянуть компиляцию).

---

## Установка (один раз)

DeepSeek-OCR грузится из **отдельного venv** `env_dsk` (transformers 4.46.3),
потому что на transformers 5.15.0 из `env/` его remote-код не импортируется
(см. выше).

```bash
# 0. Создать окружение и поставить все зависимости (один раз)
python3 -m venv env_dsk && source env_dsk/bin/activate
pip install -r approaches/DeepSeek-OCR/requirements.txt

# 1. Скачать полный снапшот (~6.7 GB): config + tokenizer + веса + remote-код
./approaches/DeepSeek-OCR/download_weights.sh
#   → approaches/DeepSeek-OCR/weights/{config.json, tokenizer*, model.safetensors,
#      modeling_deepseekocr.py, modeling_deepseekv2.py, conversation.py, ...}

# 2. Проверить импорт remote-кода
env_dsk/bin/python -c "from transformers import AutoModel, AutoTokenizer; print('ok')"
```

> Веса качаются с HuggingFace `deepseek-ai/DeepSeek-OCR` через
> `snapshot_download` (нужен весь снапшот, т.к. `trust_remote_code=True`
> требует `modeling_*.py` и `conversation.py` рядом с весами).

---

## Запуск

```bash
# 8 GB — рабочий вариант (малое разрешение, default)
env_dsk/bin/python approaches/DeepSeek-OCR/infer.py \
  /home/daniil/pdfs_test/ \
  results_DeepSeek-OCR_publaynet/ \
  --device cuda --dpi 200

# Полное разрешение Gundam (нужен GPU ≥ 12–16 GB)
env_dsk/bin/python approaches/DeepSeek-OCR/infer.py \
  /home/daniil/pdfs_test/ results_DeepSeek-OCR_publaynet/ \
  --device cuda --crop-mode --base-size 1024

# 8 GB + полное разрешение через 8-bit (экспериментально)
pip install bitsandbytes accelerate
env_dsk/bin/python approaches/DeepSeek-OCR/infer.py \
  /home/daniil/pdfs_test/ results_DeepSeek-OCR_publaynet/ \
  --device cuda --crop-mode --base-size 1024 --quantize 8bit
```

Опции:

| Аргумент | Назначение |
|---|---|
| `input_dir` | каталог с PDF и/или изображениями (`.jpg/.jpeg/.png`) |
| `results_dir` | каталог, куда пишутся JSON-результаты |
| `--model-dir` | локальный каталог со снапшотом (пропускает скачивание) |
| `--device` | `cuda` (по умолчанию); `cpu` не поддерживается (см. ниже) |
| `--dpi` | DPI рендера PDF (по умолчанию `200`) |
| `--base-size` | размер глобального вида в `model.infer()` (default `640`; полное разрешение — `1024`) |
| `--image-size` | размер кропов/вида в `model.infer()` (default `640`) |
| `--crop-mode` | включить мульти-кроп Gundam (default выкл.); на 8 GB требует `--quantize 8bit` |
| `--quantize` | `none` (bf16, default) или `8bit` (bitsandbytes, только cuda) |

> Детальный анализ железа/CUDA/памяти и порядок слоёв контейнера — в
> [`REQUIREMENTS.md`](REQUIREMENTS.md).

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
        {"bbox": [0.1, 0.05, 0.8, 0.03], "label": "title"},
        {"bbox": [0.15, 0.4, 0.6, 0.25], "label": "list"},
        {"bbox": [0.2, 0.7, 0.4, 0.2], "label": "figure"}
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

## Как устроен инференс

1. `load_model()` — `AutoModel/AutoTokenizer.from_pretrained(weights_dir,
   trust_remote_code=True)`, модель в `bfloat16` на CUDA.
2. `_predict_layout_text()` — страница сохраняется во временный PNG и
   прогоняется через `model.infer(..., eval_mode=True)`, который сам делает
   предобработку Gundam (кропы 640×640 + глобальный вид 1024×1024) и возвращает
   сырой grounding-текст (спецтокены сохранены).
3. `parse_layout_output()` — разбирает `<|ref|>CLASS<|/ref|><|det|>[[x,y,x,y]]
   <|/det|>`, нормализует 0–999 → `[x, y, w, h]` и применяет пост-обработку:
   - несколько box в одном `<|ref|>` объединяются (union);
   - близкие/пересекающиеся `figure`-блоки склеиваются в **одну фигуру**
     (`_merge_figures`);
   - идущие подряд строки `list` с выровненной левой границей склеиваются в
     **весь список** (`_merge_lists`);
   - регионы сортируются в порядке чтения (сверху вниз, слева направо).
4. Неизвестные классы (вне `LABEL_ALIASES`) молча отбрасываются, чтобы формат
   выхода совпадал с PubLayNet.

## Известные нюансы

- **Только CUDA.** `model.infer()` внутри строит bf16-CUDA-тензоры и вызывает
  `.cuda()`; `--device cpu` в `infer.py` намеренно завершается с ошибкой.
- **VRAM 8 GB — работает на малом разрешении.** Default
  `crop_mode=False, base_size=640` проверен и проходит; полное разрешение
  (`--crop-mode --base-size 1024`) на 8 GB падает с OOM — нужен GPU ≥ 12–16 GB
  или `--quantize 8bit` (bitsandbytes, экспериментально).
- **Загрузка в bf16.** Модель грузится `torch_dtype=torch.bfloat16` и
  переносится на GPU **после** каста, иначе fp32-копия (~13.3 GB) не влезает.
- **Версия transformers.** Remote-код модели написан под 4.46.3; на 5.15.0 из
  `env/` импорт падает (`LlamaFlashAttention2` отсутствует) — использовать
  отдельный venv `env_dsk` (см. раздел установки).
- Результат VLM не полностью детерминирован; используется жадный декодинг
  (`temperature=0.0` внутри `infer()`), чтобы минимизировать разброс.
- Промпт легко менять: константа `LAYOUT_PROMPT` в начале `infer.py`. Формат
  ответа модели фиксирован обучением (`<|ref|>`/`<|det|>`), поэтому в промпте
  меняются только **список классов и их определения**, но не разметка ответа.

## Ссылки

- Репозиторий: [deepseek-ai/DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR)
- Веса: [deepseek-ai/DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
- Статья: [DeepSeek-OCR: Contexts Optical Compression (arXiv:2510.18234)](https://arxiv.org/abs/2510.18234)
