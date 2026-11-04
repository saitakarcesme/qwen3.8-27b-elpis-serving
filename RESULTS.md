# Elpis Dual 262K: ölçüm sonuçları

Native ölçümler aynı 262.000 token girdi ve 128 token çıktı bütçesiyle yapıldı. Önceki motor 3,5 bpw + Q4 KV + MTP, Elpis 4 bpw + Q3 KV + DFlash2 kullanıyor. Bu bir deployment karşılaştırmasıdır; ağırlık ve cache hassasiyetleri aynı değildir. Önceki motor 123, Elpis 128 çıktı tokenı bildirdi.

| Kurulum | İlk token (sn) | Decode (token/sn) | Toplam (sn) |
|---|---:|---:|---:|
| Önceki LLMRig / TP2 MTP | 423.83 | 50.04 | 426.27 |
| Elpis tek GPU / önceki kontrollü karşılaştırma | 292.60 | 92.14 | 293.98 |
| Elpis GPU0 / 350 W | 294.20 | 91.45 | 295.59 |
| Elpis GPU1 / 420 W | 265.15 | 97.27 | 266.46 |

## İki ayrı büyük istek

Üretim API üzerinden, her biri yaklaşık 262.000 girdi tokenı + 128 çıktı tokenı olan iki istek:

- Seri toplam: **560.02 saniye**.
- Paralel tekrarlar: **295.12, 295.51 saniye**.
- Seri süre / paralel sürelerin ortalaması: **1.90× kapasite**.

Her istek ayrı motor kullandı. Girdinin başındaki farklı nonce cache hit'i engelledi; API'nin cached_tokens sayacı bütün bu isteklerde sıfırdı. Seri karşılaştırmada da iki GPU sırayla kullanıldı. Bu oran tek isteğin iki GPU üzerinde hızlanması değildir. İki bağımsız iş varsa uygulanır; tek işin builder/verifier bağımlılıkları devam eder.

## RAM ve uyumluluk

Ek 2 GiB CPU KV tier A/B/A deneyinde 308 sayfa geri yükledi, fakat TTFT 294,72 → 294,77 saniye oldu. Ek tier kapalı; mevcut recurrent-state cache üst sınırı motor başına 4 GiB. Ölçüm penceresinde host RAM baskısı ve swap hareketi görüldü; ayrıntı production-memory-summary.json içindedir. Swap için hız kazanımı iddiası yoktur.

261.862 tokenlık girdi üzerindeki üç kayıt doğru araç argümanlarıyla bulundu; araç sonucu sonrası devam isteği doğru ACK döndürdü. Gerçek Contest LocalModel istemcisiyle maksimum-context araç isteği ayrıca geçti. Gerçek MediaVerifier, görseli okuyup ikinci GPU'nun Elpis'e geri dönmesini doğruladı; geçiş 26,04 saniye sürdü. Birinci motorun sağlığı kesilmedi.

64 Contest testi ve 6 router testi geçti. Bunlar genel model kalitesinin tüm görevlerde değişmediğini kanıtlamaz. Kaynak HyperQwen incelendi; bu makinede HyperQwen runtime benchmark'ı yapılmadı. Desteklenmeyen target/draft cihaz ayırma denemesi ve kararsız küçük kazanç veren 4096 chunk üretime alınmadı.

Ham kayıtlar: /home/isa/Experiments/elpis-dual-262k-20261029/results. Araştırma: RESEARCH.md. İşletim ve geri dönüş: README.md.
