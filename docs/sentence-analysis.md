# Cümle bazlı feature ve skor akışı

```python
from app.transcript_parser import parse_transcript_document
from app.features import extract_sentence_features
from app.scoring import score_sentence_features

document = parse_transcript_document(
    "Arabulucu: Merhaba. Nasıl ilerleyelim? "
    "Sizi anlıyorum ama kabul etmelisiniz."
)
mediator_ids = [key for key, speaker in document.speakers.items()
                if speaker.role == "mediator"]
features = extract_sentence_features(
    document.sentences(), speaker_ids=mediator_ids,
)
result = score_sentence_features(features)
print(result.overall.category_totals)
print(result.overall.support_points)  # 2
print(result.overall.review_points)   # 1
```

## Hangi sonuçlar hesaplanıyor?

Parser cümleleri sıralar; features katmanı tekrar bölmez veya turları birleştirmez.
Her cümle `SentenceFeatures` içinde kaynak ID'leri, metni, karakter konumları,
eşleşen ifadeler ve isteğe bağlı sentiment sonucu ile saklanır.

| Kategori | Bir cümledeki puan | Toplandığı grup |
| --- | --- | --- |
| Empati ifadesi | En az bir eşleşme varsa 1 | Destekleyici |
| Açık uçlu soru | Başlangıç kalıbı + soru işareti varsa 1 | Destekleyici |
| Katılım daveti | En az bir eşleşme varsa 1 | Destekleyici |
| Değerlendirici ifade | En az bir eşleşme varsa 1 | İncelenecek |
| Yönlendirici ifade | En az bir eşleşme varsa 1 | İncelenecek |

Aynı cümlede birden fazla kategori bulunabilir. Tek kategorideki tekrarlar
puanı artırmaz. Kategori toplamları ve seçili cümleler içindeki yüzdeleri ayrı
verilir. Destekleyici puan cümle başına 0–3, incelenecek puan 0–2 olabilir;
birbirinden çıkarılmaz. Ham toplamlar uzunlukla artar; yüzdeler kullanım
oranıdır, iletişim kalitesi veya tarafsızlık notu değildir. Boş seçimde yüzdeler
`None`, incelenen ama eşleşmeyen cümlelerde sıfırdır.

`result.sentences` cümle sonuçlarını, `result.overall` seçili cümlelerin toplamını,
`result.by_speaker` oturum ve konuşmacı ID'sine göre özetleri verir. Aynı adı taşıyan
kişiler birleştirilmez. Bilinmeyen konuşmacılar atanmamış cümleler olarak tutulur;
yeni bir kişi çıkarımı yapılmaz. Varsayılan `speaker_ids=None` tüm konuşmacıları,
boş liste hiçbirini seçer. Arabulucu incelemesi için ID filtresi verilmelidir.

Çözüm kelimeleri ve bağlaçlar açıklayıcı veri olarak tutulur. Çözüm kelimesi
toplamı, kelimenin geçtiği cümle sayısıdır; anlaşma sağlandığını göstermez.

## Sentiment

`extract_sentence_features(..., sentiment_analyzer=analyze_sentiment)` mevcut
VADER hattını her cümle için bir kez çağırır. Varsayılan olarak kapalıdır; model
çalıştırılmaması nötr skorla temsil edilmez. `sentence.text` kullanılır, stopword
filtreli tokenlar kullanılmaz. Compound toplamı, compound ortalaması ve
olumlu/nötr/olumsuz bileşen ortalamaları ayrı döner. Ortalamaların paydası sentiment
sonucu bulunan cümle sayısıdır. Sentiment, iletişim puanlarına eklenmez.

VADER bu projede Türkçe için doğrulanmamıştır. Aynı `SentimentResult` sözleşmesini
döndüren bir fonksiyonla ileride farklı bir model bağlanabilir. İlk VADER
kullanımında mevcut sözlük indirme davranışı korunmuştur. İfade analizinin
çalışması için spaCy, NLTK veya model indirmesi gerekmez.

## Sınırlar ve uyumluluk

Bu aşama mevcut Türkçe kalıpları cümle bazına taşır. Kelime sınırları kontrol
edilir; alıntı, ironi, bağlam, olumsuzlama ve Türkçe ek çözümlemesi yapılmaz.
Örneğin alıntılanan “haklısınız” da inceleme bulgusu oluşturabilir. Hitapla başlayan
bazı sorular başlangıç kuralına uymayabilir. Bulgular eşleşen ifadeleri ve tam
cümleyi gösterir; bunlar model tarafından doğrulanmış davranış etiketleri değildir.

Eski `extract_text_features`, `scoring` ve tur bazlı skor fonksiyonları eski
çağıranlar için korunmuştur; yeni dashboard bunları kullanmaz. Eski İngilizce
hece/okunabilirlik hesapları yeni puanlamaya dahil edilmez. Aktif dinleme,
yeniden çerçeveleme ve süreç yönetimi için bu aşamada puan üretilmez.

```bash
python -m unittest discover -s tests -v
streamlit run app/app.py
```
