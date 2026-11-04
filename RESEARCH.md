# LLMRig için Elpis ve çift 3090 araştırması

Amaç, 262.144 token toplam context sınırını koruyarak tek isteğin gecikmesini ve aynı anda tamamlanabilen iş sayısını iyileştirmek. RAM, GPU belleği ve ikinci GPU aynı işi yapmaz: aktif attention hesabı GPU üzerinde kalır; RAM daha önce işlenmiş prefix sayfalarını tutabilir; ikinci GPU modelin bir bölümünü veya bağımsız bir isteği çalıştırabilir.

## Ölçülmüş başlangıç

29 Ekim yerel ölçümü aynı 262.000 token girdiyi ve 128 token çıktı bütçesini kullandı. Mevcut Qwen EXL3 3,5 bpw + Q4 KV + MTP, iki 3090 üzerinde 423,83 saniye TTFT ve 50,04 decode token/sn verdi. Elpis 4 bpw + Q3 KV + DFlash2, tek 3090 üzerinde 292,60 saniye TTFT ve 92,14 token/sn verdi. İstek toplamı 426,27 saniyeden 293,98 saniyeye indi. Bu, aynı hassasiyette tek ve çift GPU deneyi değildir. Eski motor 123, Elpis 128 çıktı tokenı bildirdi; gerçek sayaçlar önceki deneyin JSON kayıtlarında korunuyor.

## Elpis sınırları

Elpis, EXL3 matris çarpımı, recurrent state güncellemesi, INT8 attention, DFlash taslak doğrulaması ve token ağacı optimizasyonlarını birlikte kullanıyor. README'deki hız farklı uzunlukların geometrik ortalaması; doğrudan 262K hızı sayılamaz. INT8 attention ve quantized KV sayısal sonuçları değiştirebilir. Hız kazanımı genel doğruluk garantisi değildir. [Elpis açıklaması](https://github.com/gildrb/elpis/blob/13e8b593ea4c65f605b5fe7ac7d5ebb7a24f3c19/README.md), [ölçüm yöntemi](https://github.com/gildrb/elpis/blob/13e8b593ea4c65f605b5fe7ac7d5ebb7a24f3c19/bench/RESULTS.md).

Kurulu motorun KvCompactor sınıfı tensor-parallel cache için açıkça hata veriyor; attention katmanlarının birden fazla cihazda olması da reddediliyor. Elpis üzerine yalnız TP=2 eklemek çalışan bir recipe üretmez. Ağaç doğrulaması, KV taşıma ve cihazlar arası senkronizasyon ayrı geliştirme ve doğruluk testleri gerektiriyor. Kernel yamalarını vLLM üzerine kopyalamak da doğrudan kurulum yöntemi değil: tensör yerleşimleri ve recurrent state yönetimi farklı.

Daha dar bir deney, hedef modeli tek cihazda tutup DFlash modelini ikinci cihaza taşımak. Hedef attention cache tek cihazda kalır. Bu ilk GPU'da yer açar; fakat taslak ve doğrulama bağımlı olduğundan otomatik iki kat hız beklenmez. Açılan bellek daha büyük prefill parçaları için değerlendirilebilir. Yerleşim ve chunk değişikliklerinin çıktıları ayrıca kontrol edilmeli.

## HyperQwen bulguları

Issue 40 aynı makinede bir ve iki 3090 karşılaştırması sunuyor; ikinci kartla bazı decode ölçümleri iyileşiyor. Kartlar PCIe x8, NVLink yok; rapor açıkça CTX=long/huge ölçmediğini söylüyor. Bu, bizim 262K işimiz için hız tahmini vermez. KV havuzunun tek kart için sabitlenmiş değerini iki karta aynen uygulamanın belleği boşa bırakabileceği ise aktarılabilir bir tasarım dersi. [Kontrollü çift 3090 raporu](https://github.com/syv-ai/HyperQwen/issues/40).

Çoklu GPU belgesi P2P olmayan başka bir makinede TP yerine iki bağımsız motorun daha iyi toplam kapasite verdiğini kaydediyor. Donanım ve işletim sistemi farklı olduğundan bu bizim için hipotez. Bizim makinede nvidia-smi topo -p2p r iki yönde CNS döndürüyor; kartlar PHB yoluyla bağlı. GPU1 x4 bağlantısı ve P2P yokluğu nedeniyle her katmanda veri alışverişi isteyen yöntemler ölçülmeli. NCCL ortam değişkeni desteklenmeyen fiziksel P2P erişimini açmaz. [Çoklu GPU seçenekleri](https://github.com/syv-ai/HyperQwen/blob/ce9615c4f2309be727407918a6c979bd67968069/docs/multi-gpu.md).

Uzun context için KVarN, INT4/INT8 KV ve farklı attention backend'leri var. Profil adı native 262.144 sınırını kanıtlamıyor; bazı DFlash profilleri 245.760 veya daha düşük sınırla açılıyor. Burada 240K profilini 262K başarısı diye kabul etmiyoruz. Uygun KV havuzu, gerçek maksimum uzunluk ve o uzunlukta tamamlanan üretim birlikte doğrulanmalı. [Uzun context seçenekleri](https://github.com/syv-ai/HyperQwen/blob/ce9615c4f2309be727407918a6c979bd67968069/docs/long-context.md).

## 32 GB RAM bütçesi

Elpis CPUPageCache, GPU'dan çıkarılan tam sayfaları pinned RAM'e alır ve aynı prefix tekrar geldiğinde GPU'ya taşır. Yeni, bağımsız bir 262K girdinin ilk işlenmesini hızlandırmaz. İki motorun yükleme tepe belleği, Python/Torch belleği, masaüstü, Contest sandbox ve görsel doğrulama da RAM ister. Kaynaktaki 8 GiB cache varsayımı 32 GB makinede iki motora birden uygulanmamalı. Cache büyütme kararı cold TTFT, tekrar kullanılan prefix gecikmesi, gerçek restore sayacı ve bellek basıncıyla verilecek. Swap hızlandırma katmanı olarak kullanılmayacak.

## Contest Forge entegrasyonu

İstemci builder için 0,3 ve verifier için 0,15 sıcaklık gönderiyor; Elpis yalnız greedy 0 kabul ediyor. İstemci açıkça yapılandırılmalı, sunucu parametreleri sessizce yutmamalı. 240 saniyelik timeout, ölçülen 293 saniyelik cold prefill'den kısa. API anahtarı loopback yönlendiricide kalabilir; loglara yazılmamalı.

Worker tek işi seri işliyor. İki motor eklemek bunu paralel yapmaz. Paralellikte tek koordinatör ve tek submission yolu korunmalı; her iş ayrı SQLite bağlantısı ve atomik claim kullanmalı. İkinci bir worker başlatıp recover çağırmak çalışan işi tekrar kuyruğa alabileceğinden güvenli değil. Görsel inceleme ayrı Ollama servisiyle yapılıyor ve Elpis geçişinde kaybedilmemeli.

## Deneyler ve kabul koşulları

Yalnız maksimum context performansı ölçülecek. Önce hedef ve taslak GPU ayrımı; desteklenirse daha büyük prefill chunk; sonra iki bağımsız 262K motor ve kontrollü RAM cache deneyi. Tek istek hızı ve iki isteğin tamamlanma süresi ayrı raporlanacak. Cache kazancı cold sonucu ile karıştırılmayacak. Maksimum-context bilgi bulma ve araç çağrısı kontrolü, istemci testleri, sandbox entegrasyon testleri ve gerçek servis sağlık kontrolü geçişten önce tamamlanacak. Başarısız veya yavaş deneyler de korunacak.


## Aktarılabilir optimizasyonlar ve neden doğrudan birleştirilmedikleri

HyperQwen kaynak ağacı bu incelemede ce9615c4f2309be727407918a6c979bd67968069 sürümüne sabitlendi. Marlin W4A8, INT8 prefill, DFlash verify attention, recurrent-state checkpoint düzeltmeleri ve CPU prefix tier politikası ayrı katmanlar. EXL3 kullanan Elpis zaten aynı darboğazların bir kısmını kendi tensör düzenine uygun çekirdeklerle çözüyor; aynı isimli optimizasyonları üst üste eklemek ölçülebilir ek kazanç anlamına gelmiyor. [Yama listesi](https://github.com/syv-ai/HyperQwen/blob/ce9615c4f2309be727407918a6c979bd67968069/PATCHES.md).

HyperQwen'in isteğe bağlı embedding UVA yolu, token embedding ağırlıklarını pinned RAM'de tutarak GPU'da yer açıyor. Bu bir host bellek karşılığında VRAM kazanma tekniği; aktif attention hesabını CPU'ya taşımıyor. Bizde 262K zaten sığıyor ve 32 GB host belleği baskı altında. Bu nedenle önce ölçülen darboğazı azaltan değişiklikler seçiliyor; yeni bir UVA portunun otomatik hız kazancı olduğu varsayılmıyor.

KVarN da değerlendirildi. Kaynak repo vLLM için KV quantization backend'i sağlıyor; ağırlık quantization'ından ayrı ve hybrid modellerin yalnız full-attention cache'ini sıkıştırıyor. FP16'ya karşı yayımlanan kapasite/hız kazancı, Elpis'in zaten Q3 KV kullanan yoluna karşı ölçüm değil. Elpis'e KVarN eklemek yeni kernel ve cache-format portu gerektirir. [KVarN kaynak açıklaması](https://github.com/huawei-csl/KVarN/blob/main/README.md).

Tek istekte iki GPU için üç yol var: katmanları bölmek, her katmanın hesabını bölmek veya draft/target ayırmak. İlk yol tek token için çoğunlukla seri ilerler; ikinci yol iletişim maliyeti taşır; üçüncü yol draft/verify bağımlılığı ve cihaz uyumu gerektirir. Bağımsız motorlar ise ayrı işleri aynı anda çalıştırır. Seçim, modelin iki karta sığdırılması gerekliliğinden değil, bu makinedeki gerçek 262K ölçümünden yapılmalı.

## Tamamlanan 262K deneyleri

| Deney | İlk token | Decode | Karar |
|---|---:|---:|---|
| GPU0 bağımsız motor, 2048 chunk | 294,20 sn | 91,45 token/sn | İki motor adayı çalıştı |
| GPU1 bağımsız motor, 2048 chunk | 265,15 sn | 97,27 token/sn | İki motor adayı çalıştı |
| GPU1, 4096 chunk | 258,77 sn | 96,94 token/sn | İlk koşuda kazanç yok; üretimde 2048 korundu |
| Target GPU0 / draft GPU1 | Üretim başarısız | — | CUDA tree descriptor cihaz uyumsuzluğu; kullanılmadı |

İki kartın güç limitleri zaten farklıydı: GPU0 350 W, GPU1 420 W. Değiştirilmedi. Native iki motor koşularının decode aşamaları tam hizalı değildi; 91,45+97,27 toplamı eşzamanlı ölçülmüş sistem throughput'u değildir. Başarılı native koşulardaki 128 tokenlık çıktı hash'i ilk Elpis ölçümüyle aynı kaldı.

RAM deneyi aynı 262.000 tokenlık A girdisi, farklı B girdisi, sonra yeniden A şeklinde yapıldı. GPU page table yalnız en başta temizlendi. 2 GiB ek CPU KV cache, 445 sayfalık kapasite sağladı. Son A isteğinde gerçekten 308 sayfa geri getirildi; buna rağmen TTFT 294,72 saniyeden 294,77 saniyeye geldi. Çıktı hash'i aynıydı. Bu iş yükünde ek RAM cache ölçülebilir hız kazandırmadı.

Bu sonuç bütün RAM önbelleklerinin yararsız olduğu anlamına gelmiyor. Kurulu EXL3 Generator zaten her motorda en fazla 4 GiB recurrent-state cache'i sistem RAM'inde tutuyor. KV sayfaları ile aynı prefix'e ait recurrent checkpoint'in birlikte kullanılabilir olması gerekiyor. Sadece KV restore sayısı, prefill atlandığının kanıtı değildir. Ek 2 GiB pinned KV tier test sırasında host bellek payını daralttığı için üretimde kapatıldı; iki motorun mevcut recurrent-state cache'i ve GPU prefix cache'i korunuyor.

## API ve uygulama doğrulaması

Üretim router üzerinden iki farklı, cache-hit olmayan 262.004 tokenlık giriş ve her biri 128 token çıktı bütçesi eşzamanlı çalıştırıldı. İki istek ayrı GPU seçti ve toplam 295,12 saniyede bitti (GPU0 isteği 295,11; GPU1 isteği 267,59 saniye). Her ikisi 128 token üretti; toplam context 262.132 idi. Seri karşılaştırma 560,02 saniye sürdü. Paralel tekrar 295,51 saniyede bitti; iki paralel sürenin ortalamasına göre kapasite oranı 1,896 kat. Seri isteklerde giriş 262.005 tokendı; bu bir tokenlık nonce farkı ham kayıtlarda açıkça yer alıyor. Bütün isteklerde cached_tokens sıfırdı.

261.862 tokenlık araç isteği, metnin yüzde 25/50/90 bölgelerindeki üç kaydı doğru fonksiyon argümanlarıyla döndürdü. Araç sonucu eklenmiş 261.673 tokenlık devam isteği ACK ile doğru bitti. Bu dar bir uzun-context işlev kontrolüdür; genel model kalitesi veya bütün görevlerde eşdeğerlik iddiası değildir.

Contest gerçek MediaVerifier istemcisi, üretilmiş görseldeki LLMRIG 262K yazısını ve iki şekli doğru okudu. İkinci motorun durması, Ollama inference, unload ve Elpis dönüşü 26,04 saniye sürdü; birinci motor bütün sağlık örneklerinde açık kaldı. 64 Contest testi (sandbox entegrasyonu dahil) ve 6 router testi geçti; SSE testi ilk frame'in sunucu işi bitmeden istemciye ulaştığını da doğruladı.

## Kullanıcı açısından hızın anlamı

262K giriş ve 128 token çıkış deneyinde sürenin neredeyse tamamı prefill: önceki kurulumda ilk token 423,83 saniyede, istek 426,27 saniyede bitiyor. Elpis'te bunlar 292,60 ve 293,98 saniye. Decode hızının 50,04'ten 92,14 token/sn'ye çıkması yaklaşık 1,84 kat olsa da bütün isteğin hızlanması yaklaşık 1,45 kat; geçen süre yaklaşık yüzde 31 azalıyor. Uzun çıktı üreten işler için bu 128 tokenlık örneğin oranı doğrudan genellenmez.

İki bağımsız motor ise iki ayrı prefill'i aynı anda yapabildiği için toplam iş kapasitesini artırıyor. Aynı işin verifier adımı builder sonucunu bekliyorsa o bağımlılık sürer. İki GPU'nun bulunması tek bir işi otomatik ikiye bölmez. GPU1'in PCIe x4 yolu, model yerleştirildikten sonra bağımsız motorların her katmanda birbirleriyle veri alışverişi yapmasını gerektirmez; tensor-parallel bir motorda iletişim daha önemli olur.

Bu araştırmada HyperQwen kaynakları ve ilgili raporlar incelendi; HyperQwen runtime bu makinede kurulup ölçülmedi. Dolayısıyla burada verilen hız farkları LLMRig ve doğrulanmış Elpis denemeleri arasındadır; bütün motorlar arasında mutlak en hızlı kurulum iddiası yoktur.


## 32 GB sistemde gözlenen sınırlar

Servis testleri sırasında kısa süreli swap hareketi gözlendi; bu nedenle ilk paralel koşudan sonra seri karşılaştırma ve bir paralel tekrar daha yapıldı. 295,12 ve 295,51 saniyelik paralel sonuçlar birbirine yakın kaldı. Bu pencerede OOM veya konteyner yeniden başlatması olmadı. Ana sistemin kullanılabilir RAM'i en düşük yaklaşık 3,44 GiB olarak ölçüldü; bütün gelecekteki sandbox işlerinin aynı bellek davranışını göstereceği varsayılmıyor.

GPU0 masaüstünü de taşıyor ve ölçülen en dar VRAM payı yaklaşık 456 MiB; GPU1'de yaklaşık 1200 MiB. Görsel modelin ikinci Elpis ile aynı anda o GPU'da yüklenmemesi için rezervasyon/durdurma/geri yükleme akışı kullanılıyor. İki metin motorunun context sınırı bu akışta değişmiyor.

Gerçek Contest LocalModel istemcisi de aynı 261.862 tokenlık araç isteğini tamamladı: üç değer doğru, 70 çıktı tokenı, cached_tokens=0; süre 294,16 saniye. Worker ve scanner yeniden açık; yeni döngü/iş hatası görülmedi.
