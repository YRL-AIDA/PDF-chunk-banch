# DeepSeek-OCR — технические требования к запуску (железо, CUDA, память)

> Подход: `approaches/DeepSeek-OCR/`
> Статус: проверено на железе проекта (RTX 3050, 8 GB) — 2026-08-17
> Назначение: требования к окружению и памяти, при которых `infer.py` реально
> запускается. Дополняет `README.md` (инструкция) и `requirements.txt` (пакеты).

---

## 1. Архитектура железа (измерено на машине)

| Параметр | Значение | Вывод |
|---|---|---|
| GPU | NVIDIA GeForce RTX 3050, **8192 MiB (8 GB)** VRAM | Ampere, compute capability **8.6 (sm_86)** |
| Драйвер | 595.71.05, **CUDA 13.2** | см. раздел 2 |
| RAM | 15 GB всего, ~10 GB доступно | хватает на загрузку весов bf16 (~6.7 GB) в CPU перед переносом на GPU |
| CPU | 12 ядер | не лимитирует (инференс одиночный, жадный) |
| Диск | 915 GB, свободно ~161 GB | хватает на веса (~6.7 GB) и результаты |
| Python | 3.11.15 (pyenv) | `.python-version` |
| GPU занят десктопом | ~700 MiB (Xorg + gnome-shell + браузер) | **важно**: реально доступно ~7.3–7.4 GB, а не 8 GB |

`nvidia-smi` → `CUDA Version: 13.2`, `Driver Version: 595.71.05`.
`nvcc`/CUDA Toolkit **отсутствует и не нужен** (см. раздел 2).

---

## 2. Версии CUDA (правило совместимости)

Цепочка совместимости сверху вниз:

```
драйвер (CUDA 13.2)  ≥  runtime PyTorch (CUDA 12.4)  ≥  архитектура GPU (sm_86)
```

| Компонент | Версия | Примечание |
|---|---|---|
| Драйвер | 595.71.05 → CUDA **13.2** | задаёт верхнюю границу для runtime |
| `env/` (основной venv) | torch **2.13.0+cu130** | runtime CUDA 13.0 — ок для драйвера, но **transformers 5.15.0** не совместим с remote-кодом DeepSeek-OCR |
| `env_dsk` (venv подхода) | torch **2.6.0+cu124**, transformers **4.46.3** | runtime CUDA **12.4** — совместим и с драйвером, и с remote-кодом модели |

**Ключевые правила:**
- `nvcc` / CUDA Toolkit **не нужен**: PyTorch поставляет скомпилированные ядра под все архитектуры (включая sm_86). Достаточно, чтобы `torch.cuda.is_available() == True`.
- PyPI-сборка `torch==2.6.0` на Linux x86_64 — это **CUDA 12.4** (`+cu124`), и она работает на драйвере 13.2 (обратная совместимость).
- **transformers строго `==4.46.3`**: remote-код `modeling_deepseekocr.py`/`modeling_deepseekv2.py` написан под неё. На `transformers>=5` импорт падает с
  `ImportError: cannot import name 'LlamaFlashAttention2' from 'transformers.models.llama.modeling_llama'`.
- Драйвер менять не требуется; обновление PyTorch (например до 2.13+cu130) тоже не требуется и даже вредно (уйдёт от проверенной связки transformers 4.46.3).

---

## 3. Анализ памяти (почему «не на CPU» и почему OOM на GPU)

### 3.1. Модель

DeepSeek-OCR = MoE LLM DeepSeek-V2 (hidden 1280, 12 слоёв, 64 routed + 2 shared
эксперта) + энкодеры SAM-ViT-B и CLIP-L + MLP-проектор.

- Чекпойнт `model-00001-of-000001.safetensors`: **6 672 212 480 байт (~6.67 GB)**,
  2710 тензоров в **BF16** + метаданные.
- Число параметров: **3 336 106 240** (~3.34 млрд).

### 3.2. Почему не работает CPU

`model.infer(...)` (remote-код) **жёстко** делает:
```python
with torch.autocast("cuda", dtype=torch.bfloat16):
    output_ids = self.generate(input_ids.unsqueeze(0).cuda(), images=[...cuda()], ...)
```
— то есть строит bf16-CUDA-тензоры и вызывает `.cuda()`. На CPU это либо падает,
либо упирается в нереалистично долгий инференс MoE (~3.34 млрд параметров на 12
ядрах). Поэтому `--device cpu` в `infer.py` намеренно завершается с ошибкой.
**CPU-режим не поддерживается по дизайну; нужен GPU.**

### 3.3. Почему был OOM при загрузке на GPU

Корневая причина — порядок операций в `load_model()`:

```python
model = AutoModel.from_pretrained(...)            # без torch_dtype → веса грузятся в fp32
model = model.eval().to(device).to(dtype)         # .to("cuda") переносит fp32 (~13.3 GB) на GPU
```

`from_pretrained` без `torch_dtype` инстанцирует модель в fp32 (default dtype).
Итог: на 8 GB GPU пытался влезть **~13.3 GB** fp32-модель → `CUDA out of memory`
(в трейсе: «6.30 GiB allocated… tried to allocate 20 MiB» — процесс умер на
полпути переноса fp32-весов).

**Исправление (применено):** грузить сразу в bf16 и переносить **после** каста:

```python
model = AutoModel.from_pretrained(..., torch_dtype=torch.bfloat16)  # bf16 на CPU (~6.7 GB в RAM)
model = model.eval().to(torch.bfloat16).to(device)                  # на GPU уходит уже bf16
```

Проверено: модель грузится как `dtype=torch.bfloat16`, на GPU занимает **6.77 GB**
(allocated), свободно остаётся **~0.79 GB**.

### 3.4. Бюджет памяти при генерации (проверено)

| Конфигурация | Vision-токены | Итог |
|---|---|---|
| `crop_mode=False, base_size=640` (default) | 100 (BASE `[1,100,1280]`) | **работает**, ~0.79 GB свободно после весов |
| `crop_mode=True, base_size=1024` (полное Gundam) | 256 + 6×100 = 856 (`[1,256,1280]` + `[6,100,1280]`) | **OOM**: «tried to allocate 610 MiB… 274.50 MiB free» |

Причина OOM на полном разрешении — **пик logits**: remote-код делает
`logits = self.lm_head(hidden_states); logits = logits.float()`, то есть на
префилле материализует `[1, seq, 129280]` в **fp32**. При ~1200 токенах это
~610 MiB разовым скачком сверх весов (6.77 GB) — на 8 GB не помещается.
При 100–110 токенах (малое разрешение) пик ~190 MiB — помещается.

**Вывод по VRAM (8 GB):**
1. bf16 + малое разрешение (`crop_mode=False, base_size=640`) — **рабочий default**, без новых зависимостей.
2. bf16 + полное разрешение — **не помещается** (нужен GPU ≥ 12–16 GB).
3. 8-bit квантизация (`bitsandbytes`, веса ~3.4 GB) — даёт запас для полного разрешения, но требует `bitsandbytes`+`accelerate` и не проверена на custom-remote-коде (рекомендуется как опция, см. раздел 4).

### 3.5. Время и мощность

- Загрузка bf16-модели (CPU→GPU): ~12 с (измерено).
- Инференс одной страницы на малом разрешении: единицы секунд, жадный декодинг (`temperature=0.0`).
- GPU утилизируется только в момент прогона; между страницами простаивает. Пакетный прогон не используется (по одному изображению), что минимизирует пик памяти.
- Потребление: RTX 3050 до 115 W; инференс кратковременный — лимит по мощности не актуален.

---

## 4. Установка (что и как ставить; выбор способа)

### 4.1. Способ запуска — нативный venv (не Docker)

Причины (те же, что для DiT/MinerU):
- Данные (`/home/dataset/...`) и результаты (`results_*`) работают по путям напрямую, без проброса томов.
- Полный CUDA/Docker-образ (nvidia/cuda devel + torch) занимает 10–15 GB — избыточно, когда модель грузится через `transformers` + `trust_remote_code`.
- Docker оправдан только при переносе на другую машину/изоляции; порядок слоёв приведён в разделе 5.

### 4.2. Шаги установки (один раз)

```bash
# 0. Отдельный venv (transformers 4.46.3 несовместим с 5.15.0 из env/)
python3 -m venv env_dsk && source env_dsk/bin/activate
pip install -r approaches/DeepSeek-OCR/requirements.txt
#   torch==2.6.0 (CUDA 12.4) + torchvision 0.21.0
#   transformers==4.46.3 + tokenizers==0.20.3
#   einops, addict, easydict, tqdm, Pillow, numpy, safetensors, pdf2image, huggingface_hub

# 1. Системная зависимость для pdf2image (рендер PDF)
sudo apt install -y poppler-utils        # даёт pdftoppm (уже установлен: 24.02.0)

# 2. Веса (~6.7 GB) — полный снапшот (нужны modeling_*.py, conversation.py)
./approaches/DeepSeek-OCR/download_weights.sh
#   → approaches/DeepSeek-OCR/weights/

# 3. Проверка
env_dsk/bin/python -c "import torch, transformers; print(torch.__version__, torch.cuda.is_available(), transformers.__version__)"
#   ожидаемо: 2.6.0+cu124 True 4.46.3
```

### 4.3. Запуск

```bash
# 8 GB — рабочий вариант (малое разрешение, default)
env_dsk/bin/python approaches/DeepSeek-OCR/infer.py \
  /home/dataset/PubLayNet/micro_publaynet_4/pdfs/dev \
  results_DeepSeek_publaynet --device cuda

# Полное разрешение (нужен GPU ≥ 12–16 GB)
env_dsk/bin/python approaches/DeepSeek-OCR/infer.py \
  <input> <results> --device cuda --crop-mode --base-size 1024

# 8 GB + полное разрешение через 8-bit (экспериментально, см. ниже)
pip install bitsandbytes accelerate
env_dsk/bin/python approaches/DeepSeek-OCR/infer.py \
  <input> <results> --device cuda --crop-mode --base-size 1024 --quantize 8bit
```

### 4.4. 8-bit квантизация (опция для полного разрешения на 8 GB)

`--quantize 8bit` грузит модель через `load_in_8bit=True, device_map={"": 0}` —
веса ~3.4 GB, вся модель остаётся на GPU 0. Требует `bitsandbytes` и `accelerate`.
**Не проверена** на custom remote-коде (`DeepseekOCRForCausalLM` + encoders
SAM/CLIP), поэтому используйте только если малое разрешение недостаточно, и
сверяйте качество с bf16-вариантом.

---

## 5. Порядок слоёв контейнера (если потребуется Docker)

Слои упорядочены **от самых стабильных (низ) к самым частым изменениям (верх)**,
чтобы правка `infer.py`/промпта не пересобирала тяжёлые слои torch/transformers.
Веса (~6.7 GB) в образ **не запекаются** — монтируются томом.

```dockerfile
# 1. База + системные пакеты (меняется крайне редко)
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# 2. CUDA-рантайм не ставится отдельно: torch-колесо уже содержит ядра (sm_86).
#    (Если нужен полный образ — nvidia/cuda:12.4.1-runtime, но он не обязателен.)

# 3. PyTorch — самый тяжёлый и стабильный слой (закреплён версией)
RUN pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

# 4. Ядро VLM + зависимости remote-кода (закреплены версиями, меняются редко)
RUN pip install --no-cache-dir \
    transformers==4.46.3 tokenizers==0.20.3 \
    einops addict easydict tqdm Pillow numpy safetensors pdf2image huggingface_hub

# 5. Опционально: bitsandbytes accelerate (только для --quantize 8bit)
# RUN pip install --no-cache-dir bitsandbytes accelerate

# 6. Код подхода (меняется часто — самый верхний слой, кэш не инвалидирует 3–5)
COPY approaches/DeepSeek-OCR/ /app/approaches/DeepSeek-OCR/

# 7. Веса — НЕ копируются, монтируются томом:
#    docker run -v $(pwd)/approaches/DeepSeek-OCR/weights:/app/approaches/DeepSeek-OCR/weights \
#      -v /home/dataset:/data -v $(pwd)/results:/results \
#      --gpus all <image> \
#      python /app/approaches/DeepSeek-OCR/infer.py /data/... /results ...
```

**Правила:**
- `weights/`, `results_*/`, `chunk_cache/` — в `.dockerignore` (не запекать, не копировать).
- Том GPU: `--gpus all` (nvidia-container-toolkit).
- Обновление промпта/пост-обработки инвалидирует только слой 6 (и, при изменении зависимостей, — соответствующий слой 4–5).

---

## 6. Требования к коду (что обязан соблюдать `infer.py`)

1. **`load_model`** грузит `torch_dtype=torch.bfloat16` и переносит на `device`
   **после** каста (`.to(dtype)` → `.to(device)`). Запрещено `.to("cuda")` на
   fp32-модели.
2. **Разрешение** задаётся флагами `--base-size/--image-size/--crop-mode` и
   пробрасывается в `model.infer(...)`. Default — 8 GB-безопасный
   (`crop_mode=False, base_size=640, image_size=640`).
3. **CPU запрещён явно**: `--device cpu` → ошибка с пояснением (bf16-CUDA путь).
4. **`--quantize 8bit`** допустим только с `--device cuda`.
5. Результат — единый JSON `{pages:[{num, regions:[{bbox:[x,y,w,h], label}], metainfo}]}`,
   bbox нормализованы [0,1], классы PubLayNet: `text/title/list/table/figure`.
6. Пост-обработка (merge figures/lists, alias-маппинг) не зависит от разрешения
   и dtype — чистая функция над строкой-ответом.

---

## История версий

- **1.0.0** — первичные технические требования. Зафиксированы: железо (RTX 3050 8 GB,
  sm_86, driver 13.2), цепочка CUDA, корневая причина OOM (fp32-загрузка →
  `.to("cuda")`), исправление (bf16 до переноса), измеренный бюджет памяти
  (bf16 = 6.77 GB на GPU; малое разрешение работает, полное — OOM), инструкция
  установки и порядок слоёв контейнера.
