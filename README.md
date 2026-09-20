# mediSysAI

Arabuluculuk  metin transkriptlerini analiz ederek arabulucunun performansını puanlayan bir araştırma/demo projesi. Metin üzerinden duygu analizi ve çeşitli dilsel özellikler çıkarılır; ileride ses ve video tabanlı sinyallerle genişletilmesi planlanıyor.

## Proje yapısı

```
app/
  transcript_parser.py   "Konuşmacı: metin" formatındaki transkripti ayrıştırır
  sentiment.py            NLTK VADER ile cümle/tur bazlı duygu skoru hesaplar
  features.py              Metin özelliklerini çıkarır (bkz. aşağıda); video/ses
                           özellikleri için iskelet sınıflar (henüz uygulanmadı)
  scoring.py               Boş - genel puanlama mantığı (compute_scores) henüz yazılmadı
  app.py        Streamlit arayüzü denemesi
  sample_transcript.txt   Örnek/demo transkript
database/
  init_db.py               Boş - kalıcı depolama ileride eklenecek
scripts/
  model_trainer.py         Boş - özel model eğitimi ileride eklenecek
requirements.txt
```

## Mevcut özellikler

- **Transkript ayrıştırma**: `Speaker: text` biçimindeki satırları `Turn` nesnelerine çevirir.
- **Duygu analizi**: NLTK VADER ile `compound` / `positive` / `neutral` / `negative` skorları.
- **Metin özellikleri** (`features.py`):
  - Duygu skoru ortalaması
  - Açık uçlu soru sıklığı
  - Empati ifadesi sayısı
  - Dil karmaşıklığı (hece bazlı)
  - Söz dizimsel tutarlılık (cümle uzunluğu tutarlılığı + bağlaç kullanımı)
  - Anahtar kelime sıklığı (ör. "agreement", "compromise")
  - Okunabilirlik skoru (Flesch-Kincaid benzeri formül)

## Devam eden işler

- `scoring.py` şu anda boş.
- `VisualCueFeatures`, `ExpressionFeatures`, `VoiceToneFeatures` .
- `database/init_db.py` ve `scripts/model_trainer.py` ileride doldurulmak üzere oluşturulmuş boş dosyalar.

## Kurulum

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

NLTK VADER lexicon ilk çalıştırmada `sentiment.py` tarafından otomatik indirilir.

## Çalıştırma

```bash
# Basit dashboard, sample_transcript.txt'yi doğrudan okur
cd app
streamlit run app.py

```

.

## Örnek transkript formatı

```
Mediator: Thank you both for being here today.
Party A: I'm frustrated because ...
Party B: That's not fair, ...
```

## Yol haritası

- `scoring.py` yapılacak
- `database/init_db.py` veritabanı
- `scripts/` modeller için
- Ses tonu ve yüz ifadesi/görüntü tabanlı özelliklerin eklenmesi
