# OTTO Phase 1 Heuristic Baseline (Leakage-Safe)

Bu proje, OTTO için model kullanmadan Faz 1 retrieval + heuristic ranking baseline pipeline'ı sunar.

## Özellikler
- Tüm train/test parquet shard'larını okur (yalnızca `000.parquet` ile sınırlı değildir).
- Veriyi her shard içinde `session`, `ts` bazında deterministik sıralar.
- Sadece train verisinden:
  - global popularity,
  - event-type ağırlıklı popularity,
  - memory-aware co-visitation üretir.
- Candidate generation ve scoring ayrı modüllerdedir.
- `clicks`, `carts`, `orders` için ayrı heuristic scoring uygular.
- Her `session` ve hedef türü için unique ve sıralı top-20 `aid` üretir.
- `test_labels.parquet` sadece offline evaluation aşamasında kullanılır.

## Dosya Yapısı
- `src/otto_phase1/io_utils.py`: shard okuma/sıralama
- `src/otto_phase1/popularity.py`: train-only popularity
- `src/otto_phase1/covisitation.py`: shard/chunk tabanlı co-visitation
- `src/otto_phase1/candidates.py`: candidate generation
- `src/otto_phase1/scoring.py`: target-specific heuristic scoring
- `src/otto_phase1/predict.py`: test session prediction üretimi
- `src/otto_phase1/evaluate.py`: Recall@20 ve weighted score
- `src/otto_phase1/pipeline.py`: uçtan uca orchestration
- `scripts/run_phase1_baseline.py`: CLI entrypoint

## Kurulum
```bash
pip install -r requirements.txt
```

## Çalıştırma
```bash
python scripts/run_phase1_baseline.py \
  --data-root archive \
  --save-submission outputs/submission.csv
```

Sadece prediction üretmek (evaluation kapalı):
```bash
python scripts/run_phase1_baseline.py --no-eval
```

## Hızlı Evaluation (Pipeline Çalıştırmadan)
Mevcut bir submission CSV dosyasını yeniden değerlendirmek için (pipeline tekrar koşmadan):
```bash
python scripts/evaluate_submission.py \
  --submission-path outputs/submission.csv \
  --labels-path archive/test_labels.parquet
```

Bu script sadece submission CSV'si ve test_labels.parquet'ı okur; train, popularity, co-visitation veya prediction yeniden oluşturulmaz. (~1-2 saniye içinde tamamlanır.)

## Önemli Leakage Notu
`test_labels.parquet` yalnızca `evaluate.py` içinde okunur. Candidate generation, popularity, co-visitation ve scoring aşamalarında kullanılmaz.

## Ayarlanabilir Parametreler
CLI üzerinden örnek:
- `--session-max-events-covis`
- `--covis-topk-neighbors`
- `--covis-candidates-per-item`
- `--min-candidates-before-fallback`

Bu parametreler bellek-kalite dengesini kontrol etmek için tasarlanmıştır.
