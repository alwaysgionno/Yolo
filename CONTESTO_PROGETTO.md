# CONTESTO PROGETTO: Face Access Camera (riconoscimento facciale in tempo reale)

> **Come usare questo documento:** incollalo all'inizio di una nuova chat con Claude, oppure allegalo insieme allo ZIP o alla cartella del progetto. Così l'assistente può lavorare sul codice senza ricostruire tutta la storia. Leggilo per intero prima di rispondere.

---

## 1. Obiettivo

Applicazione desktop in Python, sviluppata in Visual Studio Code, che usa la webcam in tempo reale per riconoscere le persone registrate tramite embedding facciali. Quando riconosce una persona la saluta per nome; quando incontra un volto sconosciuto, chiede il nome e lo registra.

Il sistema è diviso in due repository che comunicano solo via MQTT:

- **`face-access-camera`** (questa): webcam, modelli AI, riconoscimento su una galleria in memoria, interfaccia. Non accede al database.
- **`face-access-server`**: broker MQTT (Mosquitto), servizio di ingest che salva gli eventi in PostgreSQL + pgvector e pubblica la galleria, chatbot AI. Contiene anche il contratto dei messaggi (`docs/mqtt-contract.md`).

## 2. Requisiti funzionali

1. **Architettura concorrente.** Cattura webcam, inferenza AI e interfaccia grafica girano in thread distinti. L'AI non deve mai bloccare il video.
2. **Pipeline di riconoscimento.** YOLOv8 rileva il volto nel fotogramma. Il ritaglio viene passato a InsightFace (ArcFace), che produce un embedding a 512 dimensioni. Il confronto avviene sulla telecamera, con la similarità coseno sugli embedding della galleria locale.
3. **Nuovi volti.** Se la similarità massima è sotto `RECOGNITION_THRESHOLD` il volto è sconosciuto. Quando lo stesso volto sconosciuto resta nell'inquadratura per più risultati di inferenza consecutivi, un overlay OpenCV chiede il nome. Poi vengono acquisiti da 5 a 10 campioni validi (`ENROLLMENT_SAMPLES`, predefinito 8).
4. **Apprendimento continuo.** I riconoscimenti con similarità sopra `CONTINUOUS_LEARNING_THRESHOLD` arricchiscono il profilo. Gli embedding vengono mediati a gruppi e inviati al server come nuovi campioni.
5. **Saluti.** "Benvenuto, Nome!" compare una sola volta, al primo riconoscimento della persona, che di norma coincide con la registrazione. "Bentornato, Nome!" compare solo quando la persona rientra nel campo visivo dopo un'assenza di almeno `WELCOME_BACK_ABSENCE_SECONDS` (predefinito 600), misurata dall'ultimo avvistamento. Ogni saluto viene pubblicato al server, che aggiorna `persons.last_access_at` e scrive una riga in `access_logs`.
6. **Dati.** La telecamera non salva nulla su disco. Registrazioni, saluti ed embedding di apprendimento continuo vengono pubblicati via MQTT (QoS 1) e salvati dal server in tempo reale.

## 3. Stack tecnologico

- Python 3.11 (3.10 supportato)
- opencv-python 4.10.0.84, numpy 1.26.4, ultralytics 8.3.0, insightface 0.7.3, onnxruntime 1.19.2, paho-mqtt 2.1.0, python-dotenv 1.0.1
- torch 2.4.1 CPU e torchvision 0.19.1, installati separatamente dall'indice PyTorch
- Broker MQTT del server (Mosquitto 2.0), raggiunto sulla rete locale
- Interfaccia interamente OpenCV, senza Tkinter

Le versioni sono in `requirements.txt`, con la spiegazione di ogni libreria.

## 4. Architettura e thread

```
 Webcam ──► CameraStream (thread "CameraStream")
              │  conserva solo l'ULTIMO frame (Condition + frame_id)
              ▼
            InferenceWorker (thread "InferenceWorker")
              │  FaceProcessor:      YOLO → ritaglio con margine → InsightFace
              │  RecognitionService: find_nearest (galleria), soglia, apprendimento
              │  GreetingPolicy:     presenza per persona, "Benvenuto" / "Bentornato"
              ▼
            FrameResult (ultimo risultato, protetto da lock) + coda dei saluti
              │
              ▼
            FaceAccessApp (thread principale)
              │  UIController:      disegno OpenCV, toast impilati, barra di stato
              │  EnrollmentSession: macchina a stati della registrazione
              │  NewFaceSelector:   scelta e conferma del volto sconosciuto (usato dalla sessione)
              └─ tasti: Q esce, ESC annulla, INVIO conferma il nome

 PersonStore (usato dal worker e dal thread principale)
   ├─ letture  ──► Gallery   (persone ed embedding in memoria)
   └─ scritture ─► MqttLink  (thread di rete di paho) ──► broker MQTT del server
                     └─ messaggi in arrivo ──► Gallery
```

Regole da rispettare:
- `imshow` e `waitKey` vanno chiamati solo dal thread principale.
- YOLO e InsightFace vengono usati solo dal worker di inferenza.
- `Gallery` è usata da tre thread (rete MQTT, inferenza, UI) ed è protetta da un lock. La matrice di ricerca viene sostituita per intero a ogni modifica: `find_nearest` legge sempre una coppia coerente (matrice, proprietari delle righe) senza trattenere il lock durante il calcolo.
- Le callback di paho girano sul thread di rete: non devono sollevare eccezioni (fermerebbero il thread) né fare lavoro lungo.
- Lo stato dei saluti (`GreetingPolicy`) è usato solo dal worker di inferenza: i saluti arrivano alla UI attraverso la coda dell'`InferenceWorker`, senza lock condivisi e senza chiamate dal thread principale.
- Il ciclo UI attende il fotogramma successivo (`CameraStream.wait_for_frame`, al massimo 50 ms): gira alla velocità della webcam e non sottrae CPU all'inferenza.
- Ogni `FrameResult` viene valutato per la registrazione una sola volta (confronto sul `frame_id`). I risultati più vecchi di 3 secondi, o arrivati mentre la webcam è scollegata, non vengono né disegnati né usati.

## 5. Pipeline di riconoscimento (per ogni fotogramma)

1. YOLO con `classes=[0]`, soglia `YOLO_CONFIDENCE` e dimensione `YOLO_IMAGE_SIZE`. Restituisce i bounding box.
2. Ritaglio del box con margine del 25%. Se il lato minore è sotto 160 px, il ritaglio viene ingrandito.
3. `InsightFace.get(ritaglio)` restituisce, per ogni volto, l'embedding normalizzato L2 (512 valori) e il riquadro, che viene riportato alle coordinate del frame originale. Da questo riquadro derivano la larghezza del volto (`face_width`) e la nitidezza (`sharpness`), cioè la varianza del Laplaciano misurata solo sul volto e non sullo sfondo.
   - Con i pesi per volti (`yolov8n-face.pt`) si tiene, per ogni ritaglio, il volto con `det_score` più alto.
   - Con il modello generico il ritaglio è una persona intera e può contenere anche i volti dei vicini: si tengono tutti, e a schermo viene mostrato il riquadro del volto.
   - Un volto che tocca un bordo interno del ritaglio è tagliato a metà e viene scartato: il suo embedding sarebbe inaffidabile.
   - Se più regioni portano allo stesso volto (IoU sopra 0.4), resta la versione con `det_score` più alto.
4. `find_nearest`: prodotto tra la matrice degli embedding della galleria (righe normalizzate) e il vettore del volto, cioè la similarità coseno, calcolata in modo esatto. Restituisce le `TOP_K_NEIGHBORS` righe più simili. Ogni corrispondenza riporta l'uid della persona (`Match.person_id`, una stringa) e i secondi trascorsi dall'ultimo saluto (`Match.last_access_age_s`, `None` se la persona non è mai stata salutata).
5. Se la migliore similarità è maggiore o uguale a `RECOGNITION_THRESHOLD`, la persona è riconosciuta. Altrimenti è sconosciuta.
6. Finché la galleria non è stata ricevuta dal server, `find_nearest` solleva `StoreError`: il risultato ha `gallery_ok = False`, i volti sono "Non verificato" e non vengono valutati per la registrazione.

## 6. Registrazione (EnrollmentSession e NewFaceSelector)

Stati: `IDLE → NAME_INPUT → CAPTURING → COMPLETED | CANCELLED`.

- **Selezione (`NewFaceSelector`):** tra i volti sconosciuti larghi almeno `MIN_FACE_WIDTH_PX` viene scelto il più vicino alla webcam (il più largo). Le persone note presenti nel fotogramma non bloccano la selezione.
- **Conferma:** il volto scelto deve ricomparire, sempre sconosciuto, per almeno 3 risultati di inferenza consecutivi (`UNKNOWN_CONFIRM_RESULTS`) e per `UNKNOWN_CONFIRM_SECONDS` (predefinito 1.5). Il volto viene riconosciuto tra un risultato e l'altro confrontando gli embedding (`find_same_face`, similarità `≥ 0.50`). Se esce, si allontana, viene riconosciuto o lascia il posto a un'altra persona, il conteggio riparte: un singolo fotogramma mosso o di profilo non fa comparire la richiesta del nome.
- **Avvio (`consider_start`):** quando il selettore conferma un volto, fuori dal periodo di snooze (`UNKNOWN_SNOOZE_SECONDS`, predefinito 60 secondi dopo un annullamento). La sessione riceve l'embedding del volto e da quel momento segue solo quello (`candidate_in`), ignorando gli altri. Nella UI il suo riquadro è arancione, "In registrazione".
- **Nome:** almeno 2 caratteri, senza duplicati (confronto senza distinzione di maiuscole, sui nomi della galleria comprese le persone disattivate). Timeout di 60 secondi. Se il volto non viene visto per 10 secondi (`CANDIDATE_LOST_TIMEOUT_S`) prima che sia stato digitato un carattere, la richiesta viene ritirata senza snooze.
- **Volto già registrato (`offer_frame`):** se il volto seguito viene riconosciuto come persona nota in 2 risultati consecutivi (`KNOWN_FACE_CONFIRMATIONS`), la registrazione viene annullata con snooze, sia durante il nome sia durante la cattura. I fotogrammi in cui risulta noto non diventano mai campioni.
- **Campione valido (`offer_frame`):** il volto seguito, non riconosciuto come persona già registrata, `face_width ≥ MIN_FACE_WIDTH_PX` (90), nitidezza `≥ MIN_SHARPNESS` (60), similarità con il primo campione `≥ 0.50` (stessa persona), similarità con gli altri campioni `≤ 0.985` (varietà), intervallo minimo di 0.4 secondi tra due campioni. Timeout di 30 secondi per la cattura.
- **Salvataggio (`PersonStore.create_person`):** genera l'uid della persona (UUID), pubblica l'evento `enrolled` con nome e campioni e aggiunge subito la persona alla galleria locale, con l'ultimo accesso a questo istante. Il server salva in una sola transazione persona, embedding (`source='enrollment'`) e riga `ENROLLED`. Al termine la UI mostra "Profilo creato. Benvenuto, Nome!". Se il server rifiuta la registrazione (nome già usato) pubblica un messaggio vuoto e la persona viene rimossa dalla galleria.
- **Dopo il salvataggio:** `FaceAccessApp` ignora i risultati di inferenza calcolati sui fotogrammi precedenti, che vedono ancora la persona come sconosciuta. Senza questo controllo la richiesta del nome ricomparirebbe subito.

## 7. Apprendimento continuo

Per ogni persona il servizio tiene in memoria un buffer di embedding con similarità `≥ CONTINUOUS_LEARNING_THRESHOLD` (0.55) e nitidezza sufficiente. Nei `CONTINUOUS_MIN_INTERVAL_SECONDS` (600 secondi) successivi a un aggiornamento il buffer non viene alimentato. Quando il buffer raggiunge `CONTINUOUS_BATCH_SIZE` (5):

1. calcola la media dei vettori;
2. la normalizza in L2;
3. la pubblica al server con l'evento `embedding` e, come qualità, la nitidezza media del gruppo.

Il server la salva con `source='continuous'`, elimina i campioni più vecchi oltre il proprio limite (`MAX_CONTINUOUS_EMBEDDINGS` nel `.env` del server), ricalcola `persons.sample_count` e ripubblica la persona: il nuovo embedding entra nella galleria quando torna dal server.

## 8. Saluti (GreetingPolicy)

`GreetingPolicy.observe(person_id, last_access_age_s)` viene chiamato dal `RecognitionService` a ogni riconoscimento e tiene in memoria `{person_id: istante dell'ultimo avvistamento}`.

- **Persona già vista in questa sessione:** conta solo l'assenza dal campo visivo. Se dall'ultimo avvistamento sono trascorsi almeno `WELCOME_BACK_ABSENCE_SECONDS` il saluto è "Bentornato", altrimenti nessun saluto. Chi resta davanti alla webcam aggiorna continuamente l'ultimo avvistamento e non viene più salutato.
- **Primo avvistamento dopo l'avvio:** fa fede l'ultimo accesso noto alla galleria (`persons.last_access_at`, ricevuto dal server e aggiornato localmente a ogni saluto). Se è `NULL` la persona non è mai stata salutata e riceve "Benvenuto". Se è più vecchio di `WELCOME_BACK_ABSENCE_SECONDS` riceve "Bentornato". Altrimenti nessun saluto.
- **Dopo la registrazione:** `create_person` valorizza l'ultimo accesso nella galleria locale, quindi la persona appena accolta con "Benvenuto" non riceve un secondo saluto al primo riconoscimento.

`RecognitionService.process` identifica prima tutti i volti e solo dopo decide i saluti: un errore a metà fotogramma non consuma saluti che non potrebbero essere mostrati. I saluti viaggiano come oggetti `Greeting(kind, full_name)`; il testo viene composto dalla UI (`_greeting_texts` in `app/main.py`), un messaggio per tipo, al plurale se le persone sono più di una ("Bentornati, Mario e Anna!").

I toast della UI sono in coda (`UIController.show_toast`): fino a 3 messaggi sono visibili impilati, gli altri attendono il loro turno, e ognuno resta a schermo per tutta la sua durata.

## 9. Galleria locale e collegamento MQTT

Il contratto completo è in `docs/mqtt-contract.md` della repository del server; `app/contract.py` ne contiene topic e codifiche. Tutti i messaggi sono JSON con QoS 1; un embedding è il base64 di 512 float32 little-endian.

- **`Gallery` (`app/gallery.py`):** copia in memoria delle persone registrate. Riceve `gallery/{uid}` (una persona con tutti i suoi embedding; messaggio vuoto = rimossa) e `gallery_info` (elenco degli uid validi). È "pronta" (`ready`, non torna più indietro) quando ha ricevuto una voce per ogni uid dell'elenco. A ogni `gallery_info` rimuove le persone non più in elenco, tranne quelle registrate localmente e non ancora confermate dal server (`_pending`). Una persona non attiva ha zero embedding: non viene riconosciuta, ma il nome resta riservato.
- **`MqttLink` (`app/mqtt_link.py`):** client paho con `connect_async` e `loop_start`: l'avvio non si blocca se il broker non risponde, e la riconnessione è automatica. Sessione non persistente: a ogni connessione si iscrive di nuovo e riceve i messaggi retained. Pubblica `camera/status` (online/offline, con last will). Gli eventi pubblicati senza connessione restano nella coda di paho (in memoria, al massimo 1000) e partono alla riconnessione. `server_online` è vero se il broker è connesso e `server/status` vale `online`.
- **`PersonStore` (`app/person_store.py`):** l'unica interfaccia usata da `RecognitionService` ed `EnrollmentSession`. `find_nearest` e `person_exists` leggono la galleria e sollevano `StoreError` se non è pronta. `create_person`, `record_greeting` e `add_continuous_embedding` pubblicano gli eventi `enrolled`, `greeted` ed `embedding`, completandoli con `event_id` (UUID, usato dal server per scartare i doppioni) e `occurred_at` (orologio della telecamera). `StoreError` anche se la coda è piena.

La telecamera non conosce lo schema del database: per quello vedi la repository del server.

## 10. Configurazione (`.env`, letto da `app/config.py`)

| Variabile | Predefinito | Significato |
|---|---|---|
| MQTT_HOST / MQTT_PORT | 127.0.0.1 / 1883 | Broker MQTT del server. Usa un indirizzo IP, non `localhost`: su Windows prova prima IPv6 e rallenta la connessione |
| MQTT_USERNAME | camera | Utente del broker (definito in `mosquitto/acl` del server) |
| MQTT_PASSWORD | (obbligatoria) | Uguale a `MQTT_CAMERA_PASSWORD` nel `.env` del server; non può essere il valore di esempio |
| CAMERA_INDEX / CAMERA_WIDTH / CAMERA_HEIGHT | 0 / 1280 / 720 | Webcam |
| YOLO_MODEL_PATH | models/yolov8n-face.pt | Pesi YOLO per i volti (con fallback su yolov8n.pt) |
| YOLO_CONFIDENCE / YOLO_IMAGE_SIZE | 0.5 / 640 | Parametri di YOLO (la dimensione deve essere un multiplo di 32) |
| INSIGHTFACE_MODEL / INSIGHTFACE_DET_SIZE | buffalo_l / 320 | Modello InsightFace e risoluzione del rilevatore interno (multiplo di 32) |
| RECOGNITION_THRESHOLD | 0.40 | Soglia di riconoscimento (similarità coseno) |
| CONTINUOUS_LEARNING_THRESHOLD | 0.55 | Soglia per l'apprendimento continuo (deve essere ≥ della soglia di riconoscimento) |
| WELCOME_BACK_ABSENCE_SECONDS | 600 | Assenza minima dal campo visivo prima di "Bentornato". Sostituisce `GREETING_COOLDOWN_SECONDS`, che non viene più letta |
| UNKNOWN_SNOOZE_SECONDS | 60 | Attesa prima di riproporre una registrazione annullata |
| UNKNOWN_CONFIRM_SECONDS | 1.5 | Tempo per cui un volto sconosciuto deve restare inquadrato prima della richiesta del nome |
| ENROLLMENT_SAMPLES | 8 | Campioni per la registrazione (5–10) |
| MIN_FACE_WIDTH_PX / MIN_SHARPNESS | 90 / 60 | Qualità minima dei campioni |
| CONTINUOUS_BATCH_SIZE / CONTINUOUS_MIN_INTERVAL_SECONDS | 5 / 600 | Parametri dell'apprendimento continuo |
| TOP_K_NEIGHBORS | 5 | Numero di vicini restituiti dalla ricerca nella galleria |

## 11. Mappa dei file

- `app/config.py`: `Settings` (dataclass immutabile), validazione, `ConfigurationError`
- `app/models.py`: `BBox`, `Match`, `FaceDetection`, `FrameResult`, `GreetingKind`, `Greeting`
- `app/camera_stream.py`: `CameraStream`, thread di cattura con riconnessione
- `app/face_processor.py`: `FaceProcessor`, pipeline YOLO + InsightFace, funzioni di ritaglio e nitidezza
- `app/gallery.py`: `Gallery`, persone ed embedding in memoria, ricerca del volto più simile
- `app/mqtt_link.py`: `MqttLink`, client MQTT, `LinkError`
- `app/person_store.py`: `PersonStore`, letture sulla galleria e scritture via MQTT, `StoreError`
- `app/contract.py`: topic MQTT, codifica di embedding e date, `ContractError`
- `app/recognition_service.py`: `RecognitionService`, soglia, saluti, apprendimento continuo
- `app/greeting_policy.py`: `GreetingPolicy`, regole di "Benvenuto" e "Bentornato"
- `app/inference_worker.py`: `InferenceWorker`, thread di inferenza, risultati, coda dei saluti
- `app/new_face_selector.py`: `NewFaceSelector`, `find_same_face`, scelta e conferma del volto sconosciuto
- `app/enrollment.py`: `EnrollmentSession`, `EnrollmentState`, registrazione del volto selezionato
- `app/ui_controller.py`: `UIController`, disegno OpenCV, coda dei toast, pannelli
- `app/main.py`: `FaceAccessApp` (composizione, ciclo UI, tasti) e `main()`
- `tests/`: test `unittest` della logica, con archivio, orologio e volti simulati in `tests/helpers.py`. Si eseguono con `python -m unittest`
- `.env.example`, `.gitignore`, `requirements.txt`, `README.md`, `.vscode/launch.json`, `models/LEGGIMI.txt`

## 12. Convenzioni di codice

- PEP 8, type hints ovunque, docstring in italiano, identificatori in inglese.
- Logging con il modulo `logging`, mai `print`.
- Errori gestiti con eccezioni dedicate: `ConfigurationError` per la configurazione, `StoreError` per galleria ed eventi, `ContractError` per i messaggi non validi.
- Testi a schermo in OpenCV senza caratteri accentati, perché il font Hershey non li supporta.
- Credenziali solo nel file `.env`, mai nel codice.

## 13. Limiti noti

- Senza pesi `yolov8n-face.pt` si usa il rilevamento di persone: un volto viene trovato solo se YOLO riconosce la persona, e ogni persona in più nel fotogramma allunga l'inferenza (circa 0.2 secondi con una o due persone, 0.7 con sei).
- Un volto già registrato, visto con similarità sotto soglia (occhiali, angolazione estrema, scarsa luce), viene trattato come sconosciuto. Se resta sotto soglia per tutta la durata della conferma e della cattura, il sistema può ancora proporre e completare una registrazione duplicata; se torna sopra soglia per due risultati consecutivi, la registrazione viene annullata.
- I nomi si digitano solo con caratteri ASCII: niente lettere accentate, per il limite del font Hershey.
- Il traffico MQTT non è cifrato: password ed embedding viaggiano in chiaro sulla rete locale.
- Gli eventi in coda vivono solo in memoria: se l'applicazione viene chiusa mentre il server non è raggiungibile vanno persi (alla chiusura attende al massimo 3 secondi). Una persona registrata in quel momento esisterebbe solo nella galleria locale e sparirebbe al riavvio.
- Finché la galleria non è arrivata dal server non si riconosce e non si registra nessuno: senza server la telecamera non può partire da zero.
- Un embedding di apprendimento continuo entra nella galleria solo dopo che il server lo ha salvato e ripubblicato.
- Una sola telecamera: client id MQTT fisso e nessun identificativo di telecamera nei topic.
- Gli avvistamenti sono in memoria: al riavvio si azzerano. Al primo avvistamento dopo l'avvio l'assenza viene stimata dall'ultimo accesso ricevuto con la galleria (`persons.last_access_at`), che registra l'ultimo saluto e non l'ultimo avvistamento: chi era stato salutato più di 10 minuti prima del riavvio riceve "Bentornato" anche se non si è mai allontanato.
- Una persona presente ma non riconosciuta per più di 10 minuti (voltata, coperta) risulta assente: al riconoscimento successivo riceve "Bentornato".
- I profili creati prima di questa versione e mai salutati hanno `last_access_at` nullo: al primo riconoscimento ricevono "Benvenuto", una sola volta.
- Non c'è tracking tra fotogrammi: i volti vengono riallineati tra un risultato e l'altro confrontando gli embedding. Con più volti sconosciuti si registra una persona alla volta, partendo dalla più vicina.
- Lo snooze dopo un annullamento è globale: per `UNKNOWN_SNOOZE_SECONDS` non viene proposta alcuna registrazione, nemmeno a un volto diverso.
- Su CPU con 1280x720 l'inferenza gira a pochi fps, mentre il video resta fluido.
- Stato delle verifiche (8 ottobre 2026, Windows 11, Python 3.11.9, CPU):
  - la logica è coperta dai test in `tests/`;
  - galleria, collegamento MQTT e archivio (`Gallery`, `MqttLink`, `PersonStore`) sono stati provati contro il server reale in Docker (Mosquitto 2.0.22, ingest, PostgreSQL 16): galleria iniziale, registrazione, saluto, apprendimento continuo, nome doppio rifiutato, cancellazione e disattivazione dal database, ingest fermo e riavviato, broker fermo e riavviato, riavvio della telecamera, chiusura con e senza broker. La similarità calcolata dalla galleria coincide con quella di pgvector;
  - YOLO (modello generico) e InsightFace sono stati provati sulle immagini di esempio delle due librerie;
  - l'applicazione completa (finestra, tasti, thread, registrazione, riconoscimento, saluto, apprendimento continuo, caduta e ripresa della sorgente video) era stata provata con una sorgente video simulata prima del passaggio a MQTT (circa 30 fps di video e 8 fps di inferenza): **dopo il passaggio non è stata ripetuta**;
  - le regole dei saluti ("Benvenuto" / "Bentornato"), la selezione del volto nuovo e i toast impilati sono stati provati con i test, con il ciclo UI reale alimentato da webcam, modelli e archivio simulati; non con i modelli AI in esecuzione;
  - **non ancora provati:** telecamera e server su due computer diversi, la cattura da una webcam fisica (sul PC di prova non era presente nessuna fotocamera) e i pesi `yolov8n-face.pt`. Le soglie `MIN_SHARPNESS` e `MIN_FACE_WIDTH_PX` vanno quindi tarate con la webcam reale.
- Aspetti GDPR e normativa sul lavoro: vedi la sezione 11 del README.

## 14. Possibili evoluzioni

- Tracking tra fotogrammi (ad esempio ByteTrack o IoU) per evitare identificazioni ripetute.
- GPU con `onnxruntime-gpu` e PyTorch con CUDA.
- Interfaccia Tkinter o PySide6 al posto di OpenCV.
- TLS sul collegamento MQTT.
- Coda degli eventi su disco (cifrata), per non perderli se l'applicazione viene chiusa senza server.
- Più telecamere: identificativo nei topic e client id distinti.
- Test di integrazione automatici con un broker di prova: quelli attuali simulano il collegamento.
- Snooze per singolo volto, per non bloccare gli altri sconosciuti dopo un annullamento.

## 15. Istruzioni per l'assistente

- Rispondi in italiano. Codice e commenti in italiano, identificatori in inglese.
- Prima di modificare un file, leggilo per intero e mantieni le responsabilità dei moduli.
- Non spostare la logica OpenCV nei thread di inferenza, e non eseguire modelli AI nel thread della UI.
- Non inserire credenziali nel codice: usa `.env`.
- Una modifica ai messaggi MQTT va fatta in `app/contract.py`, in `ingest/contract.py` e `docs/mqtt-contract.md` della repository del server. Una modifica incompatibile richiede un nuovo prefisso di topic (`v2`).
- La telecamera non deve accedere al database né salvare embedding su disco.
- Se aggiungi dipendenze, aggiornale in `requirements.txt` con la versione fissata e una riga di spiegazione.
- Chiedi conferma prima di modifiche che toccano dati biometrici, conservazione dei dati, trasmissione in rete o soglie di sicurezza.
