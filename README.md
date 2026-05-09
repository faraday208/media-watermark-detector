# media-watermark-detector

> Medya dataset'lerinde watermark tespit (YOLOv8) — watermark'lı dosyaları
> rapor eder, opsiyonel olarak `/rejected`'a taşır veya siler. Şu an
> **görsel** odaklı; video desteği gelecek.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/built%20with-uv-261230)](https://github.com/astral-sh/uv)

`media-dataset-prep` pipeline'ının **04. adımı**. Standalone kullanılabilir.

---

## 🎯 Ne yapıyor?

YOLOv8 inference ile watermark tespit. Her dosya tek tek model'den geçer:

| Sonuç | Anlam |
|---|---|
| `valid: true` | Watermark **yok** (temiz dosya) |
| `valid: false` + `reason: watermark_detected (N)` | N adet watermark bbox tespit edildi |
| `valid: false` + `reason: error: ...` | YOLO inference hatası |

Her detection için **bbox + confidence** rapora yazılır (UI'da preview için).

> **Inpainting (watermark silme) yok.** Bu tool sadece **detection** yapar; pipeline scope'u "watermark'lıyı dataset'ten ayır". Inpainting (LaMa, vb.) ayrı use-case için ileride başka tool olabilir.

---

## 🚀 Kurulum

```bash
git clone https://github.com/faraday208/media-watermark-detector
cd media-watermark-detector
uv sync
```

`media-dataset-prep` workspace altında: `make install`

### Model dosyası

YOLOv8 watermark detection model gerekli:

```
~/Models/watermarks_yolov8/watermarks_s_yolov8_v1.pt   (default)
```

Farklı path için `--model` flag.

---

## 🛠️ Kullanım — CLI

### Default tarama (sadece raporla)

```bash
uv run python run.py -i ./dataset
```

### Custom model + confidence

```bash
uv run python run.py -i ./dataset \
    --model /path/to/your_model.pt \
    --confidence 0.3
```

### Watermark'lıları taşı (undoable)

```bash
uv run python run.py -i ./dataset \
    --invalid-action move \
    --invalid-dir ./watermark_rejected
```

### Sil (irreversible — onay sorar)

```bash
uv run python run.py -i ./dataset --invalid-action delete
# Onay'sız:
uv run python run.py -i ./dataset --invalid-action delete --yes
```

### Dry-run (önizleme)

```bash
uv run python run.py -i ./dataset \
    --invalid-action move --invalid-dir ./rejected \
    --dry-run
```

### Geri al

```bash
uv run python run.py --undo ./rejected/watermark_report.json
```

---

## 📋 Operation modes — özet

| Mod | Komut | Etki | Undo |
|---|---|---|---|
| **Sadece rapor** | `--invalid-action none` (default) | Dokunulmaz | – |
| **Move** | `--invalid-action move --invalid-dir D` | Watermark'lılar D'ye taşınır | ✓ |
| **Delete** | `--invalid-action delete` | Silinir | ✗ irreversible |
| **Dry-run** | + `--dry-run` | Rapor üretilir, fiziksel değişiklik yok | – |
| **Undo** | `--undo REPORT` | move-action geri alınır | – |

---

## 🚩 Tüm CLI flag'leri

| Flag | Tip | Default | Açıklama |
|---|---|---|---|
| `-i, --input` | str | – | Input klasörü (zorunlu, `--undo` hariç) |
| `-o, --output` | str | `<input>/watermark_report.json` | Rapor JSON |
| `--recursive` / `--no-recursive` | flag | True | Alt klasör tarama |
| `--model PATH` | str | `~/Models/watermarks_yolov8/...` | YOLO model dosyası |
| `--confidence` | float | 0.25 | Detection confidence eşiği |
| `--limit N` | int | 0 | Max dosya |
| `--invalid-action` | `none\|move\|delete` | `none` | Watermark'lı için aksiyon |
| `--invalid-dir` | str | – | move hedefi |
| `--dry-run` | flag | False | Simüle et |
| `--yes` | flag | False | Onay sorma (delete) |
| `--undo` | str | – | Watermark raporundan undo |

---

## ⚙️ Config

Tool config dosyası kullanmıyor; tüm parametreler CLI flag'leri. Default model path `~/Models/watermarks_yolov8/watermarks_s_yolov8_v1.pt` — kullanıcının lokal'inde.

In-process kullanımda `find_watermarks(directory, model_path=..., confidence=...)` ile direkt parametrize edilir.

---

## 🔌 In-process (library) kullanım

```python
from watermark_core import (
    find_watermarks, apply_action, undo_from_report, write_report,
)

# YOLO inference
sr = find_watermarks(
    "./dataset",
    model_path="/path/to/model.pt",
    confidence=0.25,
    recursive=True,
)
print(f"{sr.invalid_count}/{sr.total_scanned} watermark'lı")

# Aksiyon (opsiyonel)
ar = apply_action(
    sr.results,
    source_root="./dataset",
    action="move",
    invalid_dir="./rejected",
)

# Rapor + undo
write_report("./rejected/watermark_report.json",
             scan_result=sr, action_result=ar, recursive=True)
# undo_from_report("./rejected/watermark_report.json")
```

`media-dataset-prep` meta UI bu yolla in-process kullanır.

---

## 📄 Rapor formatı

```jsonc
{
  "version": "1",
  "tool": "media-watermark-detector",
  "source_root": "/abs/path",
  "recursive": true,
  "model_path": "/.../watermarks_s_yolov8_v1.pt",
  "confidence_threshold": 0.25,
  "summary": {
    "total_scanned": 100,
    "valid": 87,            // watermark'sız (temiz)
    "invalid": 12,          // watermark'lı
    "errors": 1
  },
  "action": "move",
  "invalid_dir": "/abs/.../rejected",
  "actions": [
    {"original": "/abs/.../bad.jpg",
     "moved_to": "/abs/rejected/bad.jpg",
     "reason": "watermark_detected (2)",
     "detection_count": 2}
  ],
  "skipped": 0,
  "results": [
    {"valid": false, "reason": "watermark_detected (1)",
     "filename": "x.jpg", "path": "/abs/.../x.jpg",
     "has_watermark": true, "detection_count": 1,
     "detections": [{"confidence": 0.85,
                     "bbox": [10, 20, 100, 80], "class_id": 0}],
     "error": null}
  ]
}
```

`detections[]` UI'da bbox preview için kullanılır.

---

## 🧪 Test

```bash
uv sync --group dev
uv run pytest
```

33 test: scanner edge cases (empty dir, missing model, missing ultralytics) + actions (move/delete/undo + dry-run + irreversible) + CLI argparse + e2e.

> **Not:** Gerçek YOLO inference test'i yok — model dosyası ve ağırlıklar gerektirir. Apply_action / undo testleri sentetik `ScanResult` fixture'la (mock detection sonuçları) kapsanır.

---

## ⚠️ Limitations

- **YOLOv8 model dosyası gerekli** — repo'ya commit edilmiyor (büyük binary). Default path: `~/Models/watermarks_yolov8/watermarks_s_yolov8_v1.pt`
- **Ultralytics dependency büyük** (~500 MB ile model) — CI'da headless versiyon: `ultralytics-headless` (yoksa standart)
- `--invalid-action delete` **irreversible**
- Detection sadece tespit eder — **inpainting (silme) yapmaz**. Watermark'lı dosyalar move/delete ile dataset'ten çıkarılır
- Tek thread inference (YOLO batch'leme yok şu an); 1000+ dosyada birkaç dakika
- Recursive move'da invalid'ler **flat** olarak `invalid_dir`'e iner (alt klasör hiyerarşisi korunmaz; isim çakışması `_1`, `_2`)

---

## 🏷️ Sürüm

**v1.0.1** — pipeline integrasyonu cross-tool tutarlılık: **tree-preserving move**. Recursive scan + tree-mode dataset (00 organize çıktısı) için `--invalid-action move` artık subdir hiyerarşisini koruyor (`relative_to(source_root)` mirror). +1 regression test (34 toplam).

**v1.0.0** — clean release. `watermark-detection` → `media-watermark-detector`. Convention §uyumlu refactor:
- 6 ayrı script (detect, clean, copy, prepare, split, train) → tek `run.py` (sadece detection + cleanup)
- Inpainting (LaMa) ve training scripts'ler scope dışı silindi
- argparse + standart flag'ler (-i, -o, --recursive, --invalid-action, --undo, --dry-run, --yes, --confidence, --model)
- Sidecar JSON şeması §4 uyumlu (tool, source_root, summary, actions, results)
- Action layer (move/delete) + undo
- watermark_core/ paket adı (conventions §1)
- README 8 bölüm, MIT LICENSE, 33 test
- Kişisel training dataset (`datasets/v2_finetuned/`) repo'dan silindi

---

## 📜 Lisans

[MIT](LICENSE)
