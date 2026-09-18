# LLM-válaszok kiértékelése

Ez a csomag a beszéd-beszéd pipeline (ASR → LLM → TTS) egy futásának LLM-válaszait pontozza,
irodalomból átvett módszerekkel. Bemenet a futás által amúgy is kiírt `transcripts.yaml`,
kimenet egy olvasható riport, egy CSV, egy JSON és opcionálisan BibTeX.

Vezérelv: **a 0. szint eldönthető, minden más becslés.** A riport minden módszernél
kiírja a forrást és a validáltsági szintet, hogy a számok visszakövethetők legyenek.

Három dologra használható: egy futás kiértékelésére, két futás páros összehasonlítására, ha
azt kell eldönteni, hogy egy paraméterváltoztatás javított-e a válaszokon (lásd
[Két futás összehasonlítása](#két-futás-összehasonlítása-paraméterhangoláshoz)), illetve egy
egész futásrács egy lépésben való kiértékelésére és összevetésére (lásd
[Futásrács kiértékelése](#futásrács-kiértékelése-egy-lépésben)).

A csomag önálló: nem importál semmit a pipeline többi részéből, és kétféleképpen hívható.

## Indítás parancssorból

```powershell
cd etap-speech-assistant-mwe-main
py -m evaluation --run-dir ..\output-template\20260809_164356
```

Ez a parancs modell és internet nélkül lefut. A korábbi `py evaluate_responses.py ...`
forma is működik: az a gyökérben maradt indító ugyanezt hívja.

## Hívás függvényként

```python
from evaluation import EvaluationConfig, run_evaluation

outcome = run_evaluation(EvaluationConfig(
    run_dir="output-template/20260809_164356",
    judge_models=["qwen2.5:7b-instruct"],      # elhagyható
    progress=print))                            # elhagyható állapotjelzés

print(outcome.summary.acceptance_rate)
print(outcome.results[0].relevance["request_coverage"])
outcome.write("kimenet", emit_bibtex=True)      # csak itt ír lemezre
```

A `run_evaluation` **nem ír fájlt és nem nyomtat**; az objektumokat adja vissza, így egy
szám kiolvasásához nem kell riportot visszaparsolni. A `EvaluationConfig` mezőnevei a
kapcsolókkal egyeznek, alulvonással (`--judge-model` → `judge_models` lista). A parancssor
ugyanezt a függvényt hívja, tehát a két út nem tud elcsúszni egymástól.

## Két futás összehasonlítása (paraméterhangoláshoz)

Ha nagy mennyiségű bemenet-kimenet páron fut a modell, itemenkénti humán kiértékelés nem
reális. Erre való a **páros összehasonlítás**: ugyanazokra a bemenetekre futtatott két
konfigurációt vet össze itemenként, humán címke és bíráló modell nélkül, és megmondja,
hogy a paraméterváltoztatás javított-e.

```powershell
py -m evaluation.comparison --baseline ..\futasok\temp08 --contrast ..\futasok\temp02
```

```python
from evaluation import ComparisonConfig, compare_runs

outcome = compare_runs(ComparisonConfig(baseline="futasok/temp08",
                                        contrast=["futasok/temp02"]))
pair = outcome.pairs[0]
print([m.metric.key for m in pair.improvements])
print([m.metric.key for m in pair.regressions])   # a rontásokat is kiírja
outcome.write("kimenet")
```

Egy alapvonalhoz több változat is megadható (`--contrast` ismételhető), így egy
paramétersöprés minden pontja ugyanahhoz a referenciához mérhető.

### Miért páros

A két futás ugyanazokat a promptokat kapja, ezért az összevetés itemenként párosítható. Így
a promptok közti szóródás — ami sokszorosa a paraméter hatásának — kiesik, és jóval kisebb
mintán is kimutatható a különbség. A párosítás a felhasználói szöveg alapján történik, nem
sorrend szerint, tehát átrendezett vagy részben hibás futás esetén sem csúszik el; az
ismétlődő prompt k-adik előfordulása a másik oldal k-adik előfordulásával párosul.

### Amit a riport soronként közöl

| Oszlop | Jelentés |
| --- | --- |
| `delta` | átlagos páros különbség (változat mínusz alapvonal) a metrika saját egységében |
| `95% CI` | percentilis bootstrap intervallum a páros különbségek átlagára |
| `better/worse` | hány item mozdult jobbra, illetve rosszabbra |
| `effect` | Cliff δ a szokásos sávcímkével; bináris metrikánál üres, ott a `delta` maga a hatásméret |
| `p(adj)` | Holm-korrigált p-érték, a metrika saját családján belül (`adherence`, `response`, `runtime`) |
| `verdict` | `improved`, `degraded`, `equivalent`, `no detected change` vagy `descriptive` |

Teszt: bináris (átment/megbukott) metrikán **McNemar**, mert csak azok az itemek hordoznak
információt, amelyeknek megváltozott a verdiktjük; folytonos metrikán **Wilcoxon
előjeles rangpróba**, mert a korlátos skálák nem normálisak.

### Három dolog, ami miatt a verdikt megbízható

**Az irány deklarált, nem következtetett.** Csak ott mond javulást vagy romlást, ahol
tudható, mi a jobb. A szószám és a mondatszám `descriptive` marad: sem a hosszabb, sem a
rövidebb nem eleve jobb, ezekre a tábla nem ad verdiktet.

**A szignifikancia metrikacsaládon belül korrigált.** Húsz metrika ötszázalékos szinten
korrekció nélkül nagyjából kétharmad eséllyel ad legalább egy hamis találatot, ezért a
pontozott sorokra Holm lépcsős eljárása fut — de a három előre deklarált családon
(`adherence`, `response`, `runtime`) belül külön-külön, hogy egy minőségi állítás
bizonyítékát ne hígítsa fel a mellé mért időadatok száma. A riport családonként kiírja, hány
tesztre korrigált.

**A nagyság el van választva a kimutathatóságtól.** Nagy mintán a jelentéktelen különbség is
szignifikáns lesz, ezért minden sor hatásméretet is közöl, és ahol van előre rögzített
elhanyagolhatósági küszöb, ott ekvivalencia-teszt dönthet úgy, hogy *a változtatás nem
számít* — ez erősebb állítás, mint hogy nem mutatható ki.

### Paraméterek és kimenet

| Kapcsoló | Alapérték | Szerep |
| --- | --- | --- |
| `--baseline` | – | az alapvonalként szolgáló futás könyvtára |
| `--contrast` | – | a megváltoztatott beállítással készült futás; ismételhető |
| `--spec`, `--constraints` | – | **mindkét** futásra azonosan alkalmazva |
| `--judge-model` | – | költséges: minden futást végigbíráltat |
| `--selfcheck-samples` | 0 | önkonzisztencia mindkét futáson |
| `--alpha` | 0.05 | szignifikanciaszint |
| `--n-boot` | 2000 | bootstrap újramintázás metrikánként |
| `--no-check-metrics` | ki | ne hasonlítsa össze az egyes ellenőrzéseket külön-külön |
| `--out-dir` | `./comparison_output` | kimeneti könyvtár |

Kimenet: `comparison_report.txt`, `comparison_metrics.csv` (metrikánként egy sor, gépi
feldolgozásra) és `comparison_results.json`. A riport fejléce kiírja, mely beállítások
térnek el a két `config_used.yaml` között — és figyelmeztet, ha egyik sem, mert akkor a
különbség csak mintavételi zaj.

### Mit érdemes hangolni rajta

Modell nélkül is fut, tehát bármilyen nagy prompt-halmazon olcsó: a 0. szint ellenőrzései
eldönthetők, a lefedettség és az olvashatóság determinisztikus, ezek újrafuttatva ugyanazt
adják. A `--judge-model` bekapcsolása mindkét futást végigbíráltatja: lassú, és a bírálói
sorok öröklik a bíráló korlátait, ezért szoros összevetést önmagukban ne döntsenek el.

## Futásrács kiértékelése egy lépésben

Egy paramétersöprés nem két futás, hanem egy rács: cellánként egy futáskönyvtár, és a kérdés
sosem az, hogy „ez a futás mennyit ért el”, hanem hogy „melyik beállítás volt jobb”. A batch
mód beolvassa a teljes fát, minden futást azonos módon pontoz, megépíti azokat a
kontrasztokat, amelyekre a rács készült, és egyetlen eredménykönyvtárba írja az összes táblát.

```powershell
py -m evaluation.batch --root ..\results\text-only
```

```python
from evaluation import BatchConfig, run_batch

outcome = run_batch(BatchConfig(root="results/text-only"))
print(outcome.leaderboard()[0]["adherence_item_strict"])
outcome.write("results/evaluation_result")
```

Minden futás **egyszer** értékelődik ki, akárhány kontrasztban szerepel. Ez nemcsak munkát
spórol: így egy futás számai minden táblában ugyanazok, amit a kontrasztonkénti
újraértékelés nem garantálna.

## Teljes kimenetfa kiértékelése (`evaluation.campaign`)

Az új mérési kampány több fát ír az `outputs/` alá (ASR-only, fagyasztott átiratú
LLM-karok három független indítással, valós idejű TTFA-validáció). Ezeket egy
lépésben a kampány mód értékeli ki, és az `outputs_evaluations/` alá írja:

```powershell
py -m evaluation.campaign --outputs ..\outputs --out-dir ..\outputs_evaluations
```

Ugyanez a pipeline belépési pontjáról:

```powershell
py mwe_assistant.py --evaluate-campaign ..\outputs --eval-out-dir ..\outputs_evaluations
```

A `config/<round>/<timestamp>/` elrendezésben a kerekek **replikátumok**, nem külön
konfigurációk. A kontrasztok cellánként épülnek; a launch-szintű ICC(2,1)
(Shrout és Fleiss, 1979) és a launch-tartomány a host-stabil állításokhoz kell.
A mért és a rekonstruált TTFA egyezését Bland–Altman (1986) vizsgálja. A két ismert
hibás felvétel (`572a0bfaaf94a219006aa77a`, `5729e500af94a219006aa6b5`) alapból
kimarad.

### Az öt kontraszt

A csoportosítás a rögzített konfigurációból (`config_used.yaml`) készül, nem a könyvtárnévből,
így egy félreírt mappanév nem tud csendben hibás összehasonlítást előállítani. Kontraszt csak
ott jön létre, ahol **pontosan egy** tulajdonság tér el a futások között — ez teszi a
különbséget ahhoz a tulajdonsághoz köthetővé.

| Kontraszt | Mit tart fixen | Mi változik | Alapvonal |
| --- | --- | --- | --- |
| `parameters` | modell, kvantálás, felismerő | dekódolási beállítás (`t`, `seed`) | a greedy (`t=0`) futás |
| `quantization` | modell, méret, beállítás, felismerő | súlypontosság | a legnagyobb pontosság (`fp16`) |
| `model_size` | modellvonal, pontosság, beállítás, felismerő | paraméterszám | a legnagyobb modell |
| `cross_model` | dekódolási beállítás, felismerő | a modell | a legerősebb változat |
| `recognizer` | modell és dekódolási beállítás | a felismerő (motor, akusztikus modell, eszköz) | a rács többségében használt felismerő |

Az alapvonal nem azt állítja, hogy az a legjobb: az a **referencia, amihez a változást
mérjük**. Ezért a negatív delta úgy olvasandó, hogy *ennyibe kerül a takarékosabb
konfiguráció*. A 16 cellás rács (2 modellvonal, 6 méret, 3 kvantálás, 2 dekódolási
beállítás) ebből 16 csoportot és 32 páros összevetést ad.

A felismerőt a modellkontrasztok **fixen tartják**: amit a felismerő kiírt, az a modell
bemenete, így két felismerő összekeverése egy bemenetváltozást jelentene modellhatásként. A
felismerő azonosítója a motor mellett az akusztikus modellt, az eszközt és a számítási
pontosságot is tartalmazza (`vosk-small-en-us-0.15-cpu`, `whisper-small-cuda-int8`), mert
ugyanaz a motor más modellel másképp ismer fel. Ahol a rács több felismerőt fog át, a
csoportazonosító kiírja, melyiket tartja fixen; egyetlen felismerő esetén a nevek
változatlanok. A `recognizer` kontraszt az egyetlen, amelyben a két oldal átirata eltér: a
párosítás ezért a felvétel nevére megy, nem a felismert szövegre, különben épp azok az itemek
esnének ki, amelyeken a két felismerő nem egyezik.

A méret-létra **modellvonal** szerint csoportosul, nem pontos családnév szerint: a kapható
méretek kiadások között oszlanak el (a llama3.2 az 1b és 3b, a llama3.1 a 8b), így a
családhatárnál megálló létrából épp a legnagyobb modell maradna ki. Ahol a létra átlép egy
kiadást, azt a csoport `varying` mezője kiírja — ott a paraméterszám mellett a kiadás is
változik, és ezt nem lehet szétválasztani.

### Amit előbb ellenőriz, mint hogy összehasonlítana

A páros összevetés csak akkor a konfigurációról szól, ha minden más fixen volt. A batch ezt
nem feltételezi, hanem ellenőrzi, és minden sérülést azokkal az eredményekkel együtt ír ki,
amelyeket érint: ugyanaz az item-halmaz, ugyanaz a bemeneti szöveg, ugyanaz a rendszerprompt,
ellenőrzés-definíció, felismerő, számítási mód, kontextusablak és tokenkorlát. Az alternatíva
egy olyan különbségtábla, amiben csendben benne van egy promptcsere is.

Az ellenőrzés **kontrasztonként** fut, és a sérülést ahhoz a kontraszthoz írja ki, amelyet
érint: egy két felismerőt átfogó rács nem sérülés, ha minden modellkontraszt egy felismerőt
tart fixen. A `recognizer` kontrasztban a felismerő és az átirat eltérése maga a vizsgált
tulajdonság, ezért ott a **szándékolt kérdés** azonosságát ellenőrzi helyettük.

### Kimenet

`--out-dir` nélkül a `--root` **mellé** ír, `evaluation_result` néven: így az újraértékelés
nem ír bele abba a könyvtárba, amit olvas, és a teljes összevetés egy archiválható egység.

| Fájl | Tartalom |
| --- | --- |
| `summary_report.txt` | a rács egyetlen olvasható jelentése: bemenet, két rangsor, válasz-egyezés, strátumok, kontrasztok, forrásjegyzék |
| `leaderboard.csv` | futásonként egy sor: a konfiguráció és minden fő mérőszám egymás mellett |
| `comparison_index.csv` | minden kontraszt minden metrikája p-értékkel és verdikttel, gépi feldolgozásra |
| `input_reference.csv` | a közös bemenet felismerőnként: szándékolt szöveg, felismert szöveg, itemenkénti hibaarányok |
| `input_quality_strata.csv` | futás × bemeneti minőség szerinti bontás |
| `input_quality_impact.csv` | a felismerési hiba rangkorrelációja minden válasz-mérőszámmal |
| `all_items.csv` | minden futás minden iteme egy táblában (futásszám × n), pivotálásra vagy ábrához |
| `batch_manifest.json` | mely futásokat olvasta, milyen beállításokkal, és hova írta az eredményüket |
| `runs/<cella>/` | futásonkénti riport, itemenkénti CSV, JSON, és a `config_used.yaml` másolata |
| `comparisons/<típus>/<csoport>/` | egy kontraszt teljes páros összehasonlító táblája |

### Paraméterek

| Kapcsoló | Alapérték | Szerep |
| --- | --- | --- |
| `--root` | – | a futásokat tartalmazó könyvtár; rekurzívan keres transzkriptet |
| `--out-dir` | `<root>/../evaluation_result` | kimeneti könyvtár |
| `--group` | mind az öt | csak a megadott kontraszt épüljön meg; ismételhető |
| `--spec`, `--constraints` | – | **minden** futásra azonosan alkalmazva |
| `--answer-key` | – | a korpusz metaadat-táblája (CSV) a referencia-válaszokkal, **minden** futásra azonosan |
| `--judge-model`, `--selfcheck-samples` | – | költséges: minden futásra lefut |
| `--no-latency` | ki | ne olvassa a latencianaplókat |
| `--latency-warmup` | a futás `log_averages.json`-jában rögzített érték | hány kezdő elem maradjon ki a latenciaösszesítésből; a minőségi pontszámokat nem érinti |
| `--alpha`, `--n-boot`, `--seed` | 0.05, 2000, 0 | mint a páros összehasonlításnál |
| `--quiet` | ki | ne írja a jelentést a kimenetre |

## Függőségek

`requests`, `PyYAML`, `numpy` — telepítés: `pip install -r evaluation/requirements.txt`.
A `scipy` és a `sentence-transformers` opcionális, futásidőben észlelt. A modellt igénylő
szintekhez helyben futó Ollama kell; ha nincs, a futás nem áll le, csak jelzi a kimaradt
szintet.

## A csomag szerkezete

| Fájl | Szerep |
| --- | --- |
| `pipeline.py` | `EvaluationConfig`, `EvaluationOutcome`, `run_evaluation()` — a vezénylés |
| `cli.py`, `__main__.py` | parancssori réteg a `run_evaluation` fölött |
| `comparison.py` | két futás páros összehasonlítása, saját parancssorral |
| `batch.py` | egy futásrács kiértékelése és a kontrasztok megépítése, saját parancssorral |
| `replicates.py` | független indítások (launch) mint kísérleti egység: ICC, tartomány |
| `ttfa_validation.py` | mért vs. rekonstruált TTFA, Bland–Altman egyezés |
| `campaign.py` | teljes `outputs/` fa kiértékelése `outputs_evaluations/` alá |
| `constraints.py` | 0. szint: eldönthető ellenőrzések |
| `asr.py` | felismerési hűség a szándékolt kérdéshez képest: WER, CER, tartalmi fedés, strátumok |
| `latency.py` | itemenkénti szakaszidők és erőforrás-költség a futás latencianaplójából |
| `relevance.py` | 1. szint: lefedettség és átfedés |
| `factuality.py` | 2. szint: audit-atomok, önkonzisztencia, atomi állítások |
| `judge.py` | 3. szint: rubrika, panel, páros összevetés |
| `agreement.py`, `stats.py` | 4. szint és a statisztikai eszközök |
| `aggregation.py` | itemenkénti pontszámok, futásszintű összegzés, elfogadási politika |
| `reporting.py` | riport, CSV, JSON előállítása |
| `references.py` | módszer → irodalom → validáltsági szint nyilvántartás |
| `loaders.py` | transzkript, forgatókönyv-leírás, korpusz-válaszkulcs és futás-artefaktumok beolvasása |
| `textutils.py` | mondatvágás, normalizálás, szótagszámlálás — a mérőszámok közös alapja |
| `ollama_client.py` | HTTP-réteg a helyi modellhez: újrapróbálkozás, JSON-visszanyerés |
| `selftest.py` | 177 ellenőrzés, hálózat nélkül |
| `fake_ollama.py` | protokoll-hű teszt-szerver a modellt igénylő szintekhez |
| `requirements.txt` | a három kötelező függőség, plusz a két opcionális megjegyzésben |

Két adatkönyvtár tartozik a csomaghoz. A `rubrics/` **nem elhagyható**: a kód
alapértelmezései hivatkoznak rá.

| Fájl | Szerep |
| --- | --- |
| `rubrics/constraints_short_opener.yaml` | a 0. szint 15 ellenőrzésének definíciója (`--constraints` alapértéke) |
| `rubrics/quest_therapy_v1.yaml` | a 15 dimenziós bírálati rubrika (`--rubric` alapértéke) |

Az `examples/` ezzel szemben sablon: másold és írd át, a csomag működéséhez nem kell.

| Fájl | Szerep |
| --- | --- |
| `examples/scenarios_open_domain_smoke.yaml` | teljes forgatókönyv-leírás referencia-válaszokkal és tiltott mintákkal |
| `examples/scenarios_minimal_categories.yaml` | referencia nélküli változat: csak kategória és elvárt viselkedés |
| `examples/scenarios_therapy_template.yaml` | terápiás sablon; ez sorolja fel **minden** olvasott mezőt és a 11 kategórianevet |
| `examples/human_annotations_example.csv` | a 4. szint bemeneti formátuma: `item_id,rater_id,dimension,score` |

## Szintek

| Szint | Mit mér | Mikor fut |
| --- | --- | --- |
| 0 | prompt-megfelelés, olvashatóság | mindig, modell nélkül |
| – | felismerési hűség (WER, CER, tartalmi fedés, strátumok) | ha a transzkript tartalmaz `ori_text`-et |
| – | szakaszidők és erőforrás-költség | ha a futás mellett van `latency_log_*.csv` |
| 1 | relevancia, referencia-átfedés, válasz-egyezés | mindig; az átfedéshez és az egyezéshez `--answer-key` vagy `--spec` kell |
| 2 | ténykötelmek, hallucináció | audit-atomok mindig; a többi opcionális |
| 3 | rubrika-alapú bírálat | `--judge-model` esetén |
| 4 | értékelői egyetértés, kalibráció | `--human-annotations` esetén |

### Bemeneti minőség és futásidő

A két számozás nélküli sor nem a válaszról szól, hanem arról, hogy **mit kapott a modell** és
**mibe került a válasz**. Egyik sem igényel modellt vagy referencia-választ.

A felismerési hűség a transzkript `ori_text` mezőjéhez, vagyis a szándékolt elhangzott
kérdéshez mér. Ez nem minőségi pontszám: azt korlátozza, hogy a modell mire *tudott*
válaszolni. A számokat tartalmazó referenciánál több elhangzási változat (tőszámnév, évszám,
számjegyenkénti olvasat) közül a legjobban illeszkedő számít, hogy a felismerő ne kapjon
hibapontot azért, mert az „1990”-et kimondva hallotta. Az itemek WER alapján három
strátumba kerülnek — `clean` (hibátlan), `mild` (a referenciaszavak legfeljebb harmada
sérült), `severe` (ennél több) —, és a jelentés strátumonként is közli a prompt-megfelelést.
Lefedettséget strátumok között nem érdemes összevetni: egy zárt kérdésre adott helyes válasz
nem ismétli meg a kérdést, így a lefedettsége a modelltől független okból alacsony.

A futásidő a futás saját `latency_log_*.csv`-jéből jön, itemenként, a felvétel neve szerint
párosítva — nem sorrend szerint, hogy egy megszakadt futás se csúsztassa el az egészet. A
szakaszok átfedik egymást és nem ugyanabból a pillanatból indulnak, ezért nem összegezhetők:
a `stt` a felvétel beszédtempóban való beérkezése (fájlbemeneten a hang hossza, nem
számítási költség), a `ttfa` a beszéd végétől az első kiadott hangig tart — ez az, amit a
felhasználó kivár —, az `e2e_response_ready` pedig a teljes tétel, a felvétellel együtt,
ezért főleg a beszéd hosszát tükrözi. Modellek összevetésénél a `ttfa` és a tokenátbocsátás
a beszédes oszlop.

## Paraméterek

| Kapcsoló | Alapérték | Szerep |
| --- | --- | --- |
| `--run-dir` | – | futás könyvtára; ebből olvassa a transzkriptet, a configot és a rendszerpromptot |
| `--transcripts` | – | önálló transzkript-fájl, ha nincs meg a futás többi artefaktuma |
| `--spec` | – | forgatókönyv-leírás: kategória, referencia-válasz, kötelező/tiltott tartalom |
| `--answer-key` | – | a korpusz metaadat-táblája (CSV): `answer` / `plausible_answers` az `is_impossible` szerint |
| `--system-prompt` | futásból | felülírja a bíráláshoz használt rendszerpromptot |
| `--constraints` | `constraints_short_opener.yaml` | a 0. szint ellenőrzéseinek definíciója |
| `--max-tokens` | configból | a vágás-detektáláshoz használt `num_predict` korlát |
| `--embedding-model` | – | pl. `all-MiniLM-L6-v2`; első használatkor letöltést igényel |
| `--selfcheck-samples` | 0 | önkonzisztencia-újramintázás darabszáma; 0 = kikapcsolva |
| `--selfcheck-model` | configból | **a válaszokat előállító** modell legyen |
| `--selfcheck-temperature` | 0.8 | újramintázási hőmérséklet |
| `--fact-precision` | ki | atomi állításokra bontás és címkézés |
| `--max-claims` | 10 | állítás-korlát válaszonként |
| `--judge-model` | – | bíráló Ollama-tag; ismételhető, több modell panelt alkot |
| `--judge-url` | `localhost:11434` | Ollama végpont |
| `--judge-samples` | 1 | minta dimenziónként; 1 fölött G-Eval-szerű várható érték |
| `--rubric` | `quest_therapy_v1.yaml` | a rubrika definíciója |
| `--compare-run-dir` | – | második futás A/B összevetéshez, mindkét sorrendben |
| `--human-annotations` | – | hosszú formátumú CSV: `item_id,rater_id,dimension,score` |
| `--min-quality` | 3.5 | elfogadási küszöb a minőségi kompozitra |
| `--min-safety` | 4.0 | küszöb a legrosszabb biztonsági dimenzióra |
| `--loose-constraints` | ki | a megengedő verdikt alapján kapuz, formázási hibát elnézve |
| `--out-dir` | `<run-dir>/evaluation` | kimeneti könyvtár |
| `--emit-bibtex` | ki | BibTeX a ténylegesen használt módszerekhez |
| `--seed` | 0 | bootstrap újramintázás magja |

> Az elfogadási küszöböket **mérés előtt** kell rögzíteni: bekerülnek a riport fejlécébe,
> utólagos hangolásuk értelmetlenné teszi az elfogadási arányt.

## Kimenet

| Fájl | Tartalom |
| --- | --- |
| `evaluation_report.txt` | olvasható riport: szakaszok, forrásjegyzék, értelmezési korlátok |
| `evaluation_items.csv` | itemenként egy sor, minden mérőszámmal |
| `evaluation_results.json` | teljes strukturált kimenet |
| `evaluation_references.bib` | BibTeX, `--emit-bibtex` esetén |

## Számított értékek és értelmezésük

Az alábbi táblázatok az `evaluation_items.csv` oszlopait követik, a fájlbeli sorrendben.
Egy oszlop csak akkor kerül be, ha az őt előállító szint lefutott — ez minden szakasz
elején szerepel —, tehát egy modell nélküli futás CSV-je rövidebb, nem hiányos.

A **Státusz** oszlop jelentése: `verifiable` = konstrukció szerint eldől, `validated` =
validált eszköz vagy publikált elmélet, `established` = elterjedt, emberi korrelációval
alátámasztott módszer, `surrogate` = publikált módszer tudatos egyszerűsítése. Ahol a
forrás `–`, ott nincs mögötte publikáció: saját, de determinisztikus számítás.

### Azonosítás (mindig)

| Oszlop | Jelentés | Forrás | Státusz |
| --- | --- | --- | --- |
| `item_id` | stabil azonosító; a felvétel nevéből, a forgatókönyv-leírásból vagy sorszámból | – | verifiable |
| `filename` | a felvétel neve; ez a kapocs a latencianaplóhoz és futások között | – | verifiable |
| `category` | a forgatókönyv-kategória; ez dönti el, mely ellenőrzések és rubrika-dimenziók élnek | – | verifiable |
| `safety_critical_item` | biztonságkritikus-e az item | – | verifiable |
| `stt_text`, `llm_text` | a kiértékelt bemenet és válasz, szó szerint | – | verifiable |
| `ori_text` | a szándékolt elhangzott kérdés, ha a transzkript tartalmazza | – | verifiable |

### 0. szint – eldönthető prompt-megfelelés (mindig)

| Oszlop | Jelentés | Forrás | Státusz |
| --- | --- | --- | --- |
| `constraint_item_pass_strict` | teljesült-e **minden** kemény megszorítás; ez a fő megfelelési mutató | Zhou et al., 2023 (IFEval) | verifiable |
| `constraint_item_pass_loose` | ugyanaz, formázási átalakítások után; ha a szigorútól eltér, a hiba pusztán formázási | Zhou et al., 2023 (IFEval) | verifiable |
| `constraint_check_rate_strict` | a teljesített ellenőrzések aránya; a nem alkalmazható ellenőrzés kimarad a nevezőből | Jiang et al., 2024 (FollowBench) | verifiable |
| `constraint_check_rate_loose` | ugyanaz megengedő verdikttel | Jiang et al., 2024 (FollowBench) | verifiable |
| `constraint_failures_strict` | a bukott ellenőrzések neve, pontosvesszővel; audit-nyom | – | verifiable |
| `constraint_failures_loose` | ugyanaz megengedő verdikttel | – | verifiable |

A szigorú/megengedő pár az IFEval kettős kiértékelését követi. A 15 ellenőrzés: nem üres
válasz, nyitómondat ≤ 5 szó, nyitómondat végén pont, 0–2 részletmondat, összesen 1–3 mondat,
angol nyelv, nincs szimulált dialógus, nincs visszakérdezés (tájékoztató), befejezettség,
tokenvágás, szószám (tájékoztató), beszédidő (tájékoztató), nincs markup, valamint a
forgatókönyvből jövő kötelező és tiltott tartalom.

### Leíró és olvashatósági mérőszámok (mindig)

| Oszlop | Jelentés | Forrás | Státusz |
| --- | --- | --- | --- |
| `word_count`, `sentence_count`, `char_count` | terjedelem | – | verifiable |
| `mean_words_per_sentence` | átlagos mondathossz | – | verifiable |
| `opening_words` | az első mondat szószáma; a nyitómondat-megszorítás mért értéke | – | verifiable |
| `estimated_tokens` | **becslés**, nem mért tokenszám; a vágás gyanújához használjuk | – (tokenizáló helyett heurisztika) | surrogate |
| `estimated_spoken_seconds` | becsült beszédidő 2,5 szó/mp mellett | – (heurisztika) | surrogate |
| `flesch_reading_ease` | magasabb = könnyebb | Flesch, 1948 | validated |
| `flesch_kincaid_grade` | iskolai évfolyamban kifejezett nehézség | Kincaid et al., 1975 | validated |

### Felismerési hűség (ha van `ori_text`)

| Oszlop | Jelentés | Forrás | Státusz |
| --- | --- | --- | --- |
| `stt_wer` | szóhibaarány a szándékolt kérdéshez; a számok elhangzási változatai közül a legjobb illeszkedés számít | Levenshtein, 1966 | verifiable |
| `stt_cer` | karakterhibaarány; egy elírt szó itt kevesebbe kerül, mint a WER-ben | Levenshtein, 1966 | verifiable |
| `stt_substitutions`, `stt_deletions`, `stt_insertions` | a hibák típus szerinti bontása | Levenshtein, 1966 | verifiable |
| `stt_content_recall` | a tartalmi (nem funkció-) szavak közül mennyi élte túl a felismerést | – | verifiable |
| `stt_exact_match` | szó szerint egyezik-e a felismerés a referenciával | – | verifiable |
| `stt_stratum` | `clean`, `mild` vagy `severe` a WER alapján | Wang et al., 2003 | verifiable |

A WER nem minőségi mutató, hanem korlát: azt mondja meg, mennyire *más* kérdést kapott a
modell (Wang et al., 2003).

### Futásidő és erőforrás (ha van latencianapló)

| Oszlop | Jelentés | Forrás | Státusz |
| --- | --- | --- | --- |
| `lat_stt`, `lat_stt_endpoint_delay` | a felvétel beérkezése, illetve a beszédvég utáni lezárás; kontrollváltozók, nem a modellről szólnak | – | verifiable |
| `lat_llm_prompt_eval`, `lat_llm_ttft`, `lat_llm_ttfc` | prompt-feldolgozás, első token, első kimondható egység | – | verifiable |
| `lat_ttfa` | a beszéd végétől az első kiadott hangig: amit a felhasználó kivár | Walker et al., 1997 (PARADISE) | verifiable |
| `lat_llm_eval`, `lat_tts_total` | generálás és szintézis teljes ideje; a válasz hosszával nő | – | verifiable |
| `lat_e2e_response_ready` | a teljes tétel ideje, a felvétellel együtt | – | verifiable |
| `llm_prompt_tokens`, `llm_eval_tokens`, `llm_tokens_per_sec` | a motor saját tokenszámai és átbocsátása | – | verifiable |
| `tts_audio_ms` | a szintetizált hang hossza | – | verifiable |

A jelentés a mediánt és a 95. percentilist is közli, mert egy beszélő asszisztenst a késve
megérkező válaszok minősítenek (Dean és Barroso, 2013).

### 1. szint – relevancia (a `reference_*` oszlopokhoz `--answer-key` vagy `--spec` referencia-válasz kell)

| Oszlop | Jelentés | Forrás | Státusz |
| --- | --- | --- | --- |
| `request_coverage` | a **felismert** kérdés tartalmi szavainak IDF-súlyozott lefedettsége; referencia nem kell hozzá | – (IDF-súlyozott átfedés) | surrogate |
| `intent_coverage` | ugyanez a **szándékolt** kérdés ellen, ha van `ori_text` | – (IDF-súlyozott átfedés) | surrogate |
| `coverage_intent_gap` | a kettő különbsége: ennyit vitt el a felismerő az interakcióból | – | verifiable |
| `echo_ratio` | mennyit ismétel vissza a kérdésből; magas érték üres visszhangra utal | – | verifiable |
| `request_response_cosine` | beágyazásos hasonlóság kérdés és válasz közt; csak `--embedding-model` esetén | Zhang et al., 2020 (rokon eljárás) | surrogate |
| `intent_response_cosine` | ugyanez a szándékolt kérdés ellen; `ori_text` és `--embedding-model` kell hozzá | Zhang et al., 2020 (rokon eljárás) | surrogate |
| `answer_presence` | 0/1: benne van-e a válaszban a referencia-válasz szövege (SQuAD-normalizálás után, összefüggően) | Chen et al., 2017 | established |
| `reference_exact_match` | 0/1: a normalizált válasz *maga* a referencia-válasz | Rajpurkar et al., 2016 (SQuAD) | established |
| `reference_answer_words` | a referencia-válasz hossza szóban: e szerint válik el a rövid válasz-szakasz a bekezdés-kivonattól | – | verifiable |
| `reference_token_f1` | token-szintű átfedés a referencia-válasszal | Rajpurkar et al., 2016 (SQuAD) | established |
| `reference_rouge_1` | unigram-átfedés | Lin, 2004 | established |
| `reference_rouge_l` | leghosszabb közös részsorozat szerinti átfedés | Lin, 2004 | established |
| `reference_cosine` | beágyazásos hasonlóság a referenciához; csak `--embedding-model` esetén | Zhang et al., 2020 (rokon eljárás) | surrogate |

Több referencia-válasz esetén a legjobb egyezés kerül be. Az átfedés-alapú mérőszámok
dialógusban gyengén korrelálnak az emberi ítélettel (Liu et al., 2016), ezért **szűrésre**
valók, minőségi pontszámként nem.

Egy mondatban válaszoló asszisztensnél a három egyezés-mérőszám közül az `answer_presence` az
informatív: az `reference_exact_match` szerkezetileg nulla közeli, mert a modell nem a puszta
válasz-szakaszt adja vissza, a `reference_token_f1`-et pedig lenyomja a mondat minden további
szava. Az `answer_presence` **felső korlát** a helyességen: aki idézi a szakaszt, de közben mást
állít, kap pontot, aki más szavakkal válaszol helyesen, nem kap — graduált helyesség-ítélethez
bíráló modell (`--judge-model`) vagy emberi kör kell.

#### A korpusz saját válaszkulcsa (`--answer-key`)

A referencia-válaszok nem kézzel írt spec-ből, hanem a korpusz metaadat-táblájából (CSV)
származnak, így a `reference_*` oszlopok visszavezethetők a publikált adathalmazra. A tábla
oszlopai a SQuAD 2.0 konvencióját követik (Rajpurkar et al., 2018):

| Az item állapota | Melyik oszlop a referencia | Mit szabad rá alapozni |
| --- | --- | --- |
| `is_impossible=FALSE` | `answer`: az annotált helyes szakasz | egyezés esetén a válasz helyes volt |
| `is_impossible=TRUE` | `plausible_answers`: amit az annotátor elfogadhatónak tartott | egyezés esetén a válasz *hasonlít* arra, amit egy annotátor mondott volna — nem helyesség |

A kulcs a **felvétel neve** szerint kapcsolódik az itemekhez (item-azonosító, fájlnév, majd
kérdésszöveg sorrendben), mert a félrehallott transzkript is annak az itemnek a válaszával
mérendő, amit a rendszer kapott. Ha `--spec` és `--answer-key` is megvan, a spec az erősebb:
azt egy konkrét kiértékeléshez írták, a kulcs az egész korpuszt írja le.

A tábla két hibáját a betöltő kezeli, mindkettőt azért, mert a hallgatólagos alternatíva a
mérést rontaná: az exportáláskor a válasz körül maradt idézőjelpárt eltávolítja, a válasz-oszlopba
becsúszott bekezdés után pedig a záró idézőjelet követő szakaszt veszi válasznak. Amelyik sorban
egyik oszlopban sincs válasz, az item referencia nélkül marad, és ki sem kerül a kulcsba.

A futásösszegzés (`answer_accuracy`, illetve a `leaderboard.csv` `*_short_span`, `*_plausible`
oszlopai) három részhalmazt külön közöl, mert nem ugyanazt támasztják alá: (1) válaszolható item
rövid, legfeljebb 8 szavas válasz-szakasszal — itt a jelenlét a helyesség bizonyítéka; (2)
válaszolhatatlan item, csak elfogadható válasszal; (3) válaszolható item, de a válasz-szakasz
bekezdés-kivonat, amit néhány mondatos válasz nem tud visszaadni. A poololt arány egyiket sem
jelentené helyesen.

### 2. szint – ténykötelmek és hallucináció

Az `audit_atoms` mindig előáll. A `selfcheck_*` oszlopok `--selfcheck-samples` > 0, a
`factprecision_*` oszlopok `--fact-precision` esetén jelennek meg.

| Oszlop | Jelentés | Forrás | Státusz |
| --- | --- | --- | --- |
| `audit_atoms` | kigyűjtött évszámok, mennyiségek, nevek és darabszámuk: ellenőrzőlista emberi auditra | Ji et al., 2023 (taxonómia) | verifiable |
| `selfcheck_kernel` | `token_f1_surrogate` vagy `embedding_cosine`: melyik támogatás-mérték futott | Manakul et al., 2023 | surrogate |
| `selfcheck_samples` | hány újramintát vetett össze | Manakul et al., 2023 | established |
| `selfcheck_mean_inconsistency` | 0–1; mennyire nem támogatják a válasz mondatait az újraminták | Manakul et al., 2023 | established |
| `selfcheck_max_inconsistency` | a legrosszabb mondat értéke | Manakul et al., 2023 | established |
| `selfcheck_flagged_sentences` | a küszöb fölötti mondatok, szó szerint | Manakul et al., 2023 | established |
| `selfcheck_error` | miért maradt ki, ha kimaradt | – | verifiable |
| `factprecision_n_claims` | a megítélt atomi állítások száma | Min et al., 2023 (FActScore) | established |
| `factprecision_supported`, `factprecision_unsupported`, `factprecision_unverifiable` | a három címke darabszáma | Min et al., 2023 (FActScore) | established |
| `factprecision_precision` | a támogatott állítások aránya | Min et al., 2023 (FActScore) | established |
| `factprecision_knowledge_source` | referencia-válasz volt-e a tudásforrás, vagy a modell saját tudása | – | verifiable |
| `factprecision_unsupported_claims` | a nem támogatottnak címkézett állítások | Min et al., 2023 (FActScore) | established |
| `factprecision_error` | miért maradt ki, ha kimaradt | – | verifiable |

Két korlát: az önkonzisztencia a **bizonytalanságból** fakadó hallucinációt fogja meg, a
magabiztosan és következetesen ismételt tévedés láthatatlan marad. A támogatás-mérték
alapértelmezésben token-átfedés, nem a publikált BERTScore/NLI kernel, ezért az abszolút
értékek publikált SelfCheckGPT-számokkal nem vethetők össze — a futáson belül rangsorolnak.
Referencia nélküli `factprecision_precision` esetén a bíráló saját tudása a mérce, ezért a
`factprecision_knowledge_source` mezőt közlés előtt el kell olvasni.

### 3. szint – rubrika (csak `--judge-model` esetén)

| Oszlop | Jelentés | Forrás | Státusz |
| --- | --- | --- | --- |
| `judge_<dimenzió>` | 15 dimenzió 1–5 skálán, horgonyokkal | Liu et al., 2023 (G-Eval); Kim et al., 2024 (Prometheus 2) | established |
| `judge_<dimenzió>_panel_spread` | panel esetén a legnagyobb eltérés; nagy érték = megbízhatatlan dimenzió | Verga et al., 2024 | established |
| `quality_composite` | súlyozott átlag a **nem** biztonsági dimenziókból | Tam et al., 2024 (QUEST) | validated |
| `safety_minimum` | a legrosszabb biztonsági dimenzió; kapuként működik, nem átlagolódik | Singhal et al., 2023 | validated |

A rubrika a QUEST keretére épül, de a súlyozás helyi döntés, nem a publikált eszköz része.
A `--compare-run-dir` páros összevetésénél minden pár mindkét sorrendben lefut; a
sorrenddel változó verdikt döntetlen lesz, mert az pozíció-torzítás, nem preferencia
(Zheng et al., 2023).

### Elfogadás (mindig)

| Oszlop | Jelentés | Forrás | Státusz |
| --- | --- | --- | --- |
| `accepted` | az előre rögzített politika verdiktje | Gallifant et al., 2025 (TRIPOD-LLM) | verifiable |
| `acceptance_reasons` | miért bukott; üres, ha átment | – | verifiable |

Ha a döntéshez szükséges bemenet hiányzik — például bírálat nélküli futásban a minőségi
kompozit —, az `accepted` üresen marad, nem bukásra alapértelmezik.

### 4. szint – egyetértés és kalibráció (csak `--human-annotations` esetén)

Ezek futásszintű értékek, nem kerülnek az itemenkénti CSV-be; a riportban és a JSON-ban
találhatók.

| Mérőszám | Értelmezés | Forrás | Státusz |
| --- | --- | --- | --- |
| `percent_agreement` | nyers egyetértés az értékelőpárok közt; alapvonal | – | verifiable |
| `krippendorff_alpha_ordinal` / `_nominal` | hiányzó cellát és több értékelőt is kezel | Krippendorff, 2018 | validated |
| `cohen_kappa`, `fleiss_kappa` | véletlennel korrigált egyetértés; a sávcímkék Landis és Koch szerint | Cohen, 1960; Fleiss, 1971; Landis és Koch, 1977 | validated |
| `gwet_ac1` | aszimmetrikus kategóriákon ezt olvassuk κ helyett | Gwet, 2008; Feinstein és Cicchetti, 1990 | validated |
| `icc_2_1`, `icc_2_k` | kétszempontos véletlen modell; a címke Koo és Li szerint | Shrout és Fleiss, 1979; Koo és Li, 2016 | validated |
| `judge_bias_vs_human`, `judge_mae_vs_human` | a bíráló szisztematikus eltérése és átlagos hibája | – | verifiable |
| `judge_human_spearman`, `judge_human_kendall_tau` | rangkorreláció a bíráló és az ember közt | – (rangkorreláció) | validated |
| PPI-becslés | akkor is érvényes intervallum, ha a bíráló szisztematikusan téved | Angelopoulos et al., 2023; Boyeau et al., 2024 | validated |

### Aggregálás

Minden futásszintű átlaghoz percentilis bootstrap intervallum tartozik (Efron és
Tibshirani, 1993), ami nem igényel normalitást. Elérhető még Cliff δ (1993), TOST
ekvivalencia-teszt (Lakens, 2017) és Holm-korrekció (1979); két futás összevetésében ezekhez
jön a Wilcoxon-féle előjeles rangpróba (1945) és a McNemar-teszt (1947).

A Holm-korrekció **metrikacsaládonként** fut — `adherence`, `response`, `runtime` —, nem a
teljes táblára. A családokat a kód előre deklarálja, tehát nem az eredmény ismeretében
születnek. Ok: a mintegy negyven metrika többsége milliszekundum, és egyetlen közös
korrekció alatt egy minőségi állításhoz szükséges bizonyítékot az hígítaná fel, hogy hány
időmérés került mellé — az pedig a mérés részletessége, nem a kérdés része.

## Felhasznált irodalom

A státusz jelentése: `verifiable` = konstrukció szerint eldől, `validated` = validált eszköz
vagy publikált statisztikai elmélet, `established` = elterjedt, emberi korrelációval
alátámasztott módszer, `surrogate` = publikált módszer tudatos egyszerűsítése.

| Módszer | Forrás | Státusz |
| --- | --- | --- |
| ellenőrizhető utasításkövetés | Zhou et al., 2023 (IFEval) | verifiable |
| token-szintű F1 | Rajpurkar et al., 2016 (SQuAD) | established |
| ROUGE-1, ROUGE-L | Lin, 2004 | established |
| átfedés-metrikák korlátai | Liu et al., 2016 | validated |
| olvashatóság | Flesch, 1948; Kincaid et al., 1975 | validated |
| önkonzisztencia | Manakul et al., 2023 (SelfCheckGPT) | established |
| token-átfedéses támogatási kernel | – (a publikált BERTScore/NLI helyett) | surrogate |
| atomi ténypontosság | Min et al., 2023 (FActScore) | established |
| rubrika-alapú bírálat | Liu et al., 2023 (G-Eval); Kim et al., 2024 (Prometheus 2) | established |
| bírálói panel | Verga et al., 2024 | established |
| pozíció-torzítás kezelése | Zheng et al., 2023 | established |
| egészségügyi értékelési keret | Tam et al., 2024 (QUEST) | validated |
| empátia | Sharma et al., 2020 (EPITOME) | validated |
| túlzott elutasítás | Röttger et al., 2024 (XSTest) | established |
| klinikai túllépés | Singhal et al., 2023 | validated |
| Krippendorff α | Krippendorff, 2018 | validated |
| Cohen κ, Fleiss κ | Cohen, 1960; Fleiss, 1971 | validated |
| Gwet AC1, prevalencia-paradoxon | Gwet, 2008; Feinstein és Cicchetti, 1990 | validated |
| ICC | Shrout és Fleiss, 1979; Koo és Li, 2016 | validated |
| prediction-powered inference | Angelopoulos et al., 2023 | validated |
| bootstrap intervallum | Efron és Tibshirani, 1993 | validated |
| Cliff δ, TOST, Holm | Cliff, 1993; Lakens, 2017; Holm, 1979 | validated |
| páros rangpróba, páros bináris teszt | Wilcoxon, 1945; McNemar, 1947 | validated |
| szerkesztési távolság, szó- és karakterhibaarány | Levenshtein, 1966 | verifiable |
| a WER és a megértés kapcsolatának korlátai | Wang et al., 2003 | established |
| a 95. percentilis mint a felhasználó által érzékelt késés | Dean és Barroso, 2013 | established |
| beszélő dialógusrendszerek költség-mérése | Walker et al., 1997 (PARADISE) | validated |

A `--emit-bibtex` a ténylegesen használt módszerekhez ír BibTeX-tételeket, így a
bibliográfia nem hízik fel nem hivatkozott tételekkel.

## Korlátok, amiket a cikkben is jelezni kell

A bírálói pontszám **szűrőeszköz, nem validált mérőműszer**; humán kalibráció nélkül ne
közöljük minőségi becslésként. Az önkonzisztencia értékei a saját futáson belül
rangsorolnak, publikált SelfCheckGPT-számokkal nem vethetők össze. Az olvashatósági
formulák írott szövegre validáltak, a szintetizált beszéd érthetőségéhez hallgatási teszt
kell (ITU-T P.85 vagy P.808). Végül a lefedettség behatárol mindent: nyílt tartományú
itemeken számolt érték nem hordoz állítást terápiás forgatókönyvekről.

## Önteszt

```powershell
py -m evaluation.selftest
```

125 ellenőrzés, hálózat nélkül. A statisztikákat kézzel levezetett vagy publikált
példaértékekhez méri, mert egy hibás együttható is hihető számot ad.
