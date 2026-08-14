# Publaynet

Publaynet — это датасет для анализа макетов документов (layout analysis), содержащий PDF-файлы научных статей.

## Как скачать датасет

- Скачайте PDF-файлы Publaynet с официального источника: https://github.com/ibm-aur-nlp/PubLayNet
- Поместите все PDF в одну директорию, например `data/publaynet/pdfs/`

## Как собрать кеш чанков

```bash
python datasets/publaynet/build_chunk_cache.py data/publaynet/pdfs/ chunk_cache/
```

- Первый аргумент — путь к директории с PDF-файлами
- Второй аргумент — путь к директории, куда сохранятся JSON-файлы с чанками (рекомендуемое имя: `chunk_cache/`, уже добавлено в `.gitignore`)

## Структура кеша

- Для каждого `foo.pdf` создаётся `chunk_cache/foo.json`
- JSON содержит массив страниц, для каждой страницы — чанки (текст + bbox) и изображения (base64)

## Формат JSON

```json
{
  "pages": [
    {
      "num": 1,
      "chunks": [
        {"bbox": [0.1, 0.05, 0.8, 0.03], "text": "Introduction"}
      ],
      "images": [
        {"bbox": [0.1, 0.2, 0.5, 0.3], "base64": "..."}
      ],
      "metainfo": {"height": 792.0, "width": 612.0}
    }
  ]
}
```
