# CONTESTO PROGETTO: Face Access (riconoscimento facciale in tempo reale)

> **Come usare questo documento:** incollalo all'inizio di una nuova chat con Claude, oppure allegalo insieme allo ZIP o alla cartella del progetto. Così l'assistente può lavorare sul codice senza ricostruire tutta la storia. Leggilo per intero prima di rispondere.

---

## 1. Obiettivo

Applicazione desktop in Python, sviluppata in Visual Studio Code, che usa la webcam in tempo reale per riconoscere le persone registrate tramite embedding facciali. Quando riconosce una persona la saluta per nome, registra l'accesso nel database e aggiorna il suo profilo nel tempo. Quando incontra un volto sconosciuto, chiede il nome e lo registra. Il database è PostgreSQL con estensione pgvector, avviato con Docker.

## 2. Requisiti funzionali

1. **Architettura concorrente.** Cattura webcam, inferenza AI e interfaccia grafica girano in thread distinti. L'AI non deve mai bloccare il video.
2. **Pipeline di riconoscimento.** YOLOv8 rileva il volto nel fotogramma. Il ritaglio viene passato a InsightFace (ArcFace), che produce un embedding a 512 dimensioni. Il confronto avviene in PostgreSQL con pgvector, usando la distanza coseno.
3. **Nuovi volti.** Se la similarità massima è sotto `RECOGNITION_THRESHOLD`, un overlay OpenCV chiede il nome. Poi vengono acquisiti da 5 a 10 campioni validi (`ENROLLMENT_SAMPLES`, predefinito 8).
4. **Apprendimento continuo.** I riconoscimenti con similarità sopra `CONTINUOUS_LEARNING_THRESHOLD` arricchiscono il profilo. Gli embedding vengono mediati a gruppi e salvati come nuovi campioni.
5. **Saluto e debouncing.** Viene mostrato "Bentornato, Nome!". Ogni persona viene salutata al massimo una volta ogni `GREETING_COOLDOWN_SECONDS` (predefinito 300). Ogni saluto aggiorna `persons.last_access_at` e scrive una riga in `access_logs`.
6. **Database.** Tabelle `persons`, `face_embeddings` (VECTOR(512), indice HNSW coseno) e `access_logs`. Avvio con `docker-compose.yml`. Ispezione con DBeaver.

## 3. Stack tecnologico

- Python 3.11 (3.10 supportato)
- opencv-python 4.10.0.84, numpy 1.26.4, ultralytics 8.3.0, insightface 0.7.3, onnxruntime 1.19.2, psycopg2-binary 2.9.9, python-dotenv 1.0.1
- torch 2.4.1 CPU e torchvision 0.19.1, installati separatamente dall'indice PyTorch
- PostgreSQL 16 con pgvector (immagine Docker `pgvector/pgvector:pg16`)
- Interfaccia interamente OpenCV, senza Tkinter

Le versioni sono in `requirements.txt`, con la spiegazione di ogni libreria.

## 4. Architettura e thread

```
 Webcam ──► CameraStream (thread "CameraStream")
              │  conserva solo l'ULTIMO frame (Condition + frame_id)
              ▼
            InferenceWorker (thread "InferenceWorker")
              │  FaceProcessor:      YOLO → ritaglio con margine → InsightFace
              │  RecognitionService: find_nearest (pgvector), soglia, saluto, apprendimento
              ▼
            FrameResult (ultimo risultato, protetto da lock) + coda dei saluti
              │
              ▼
            FaceAccessApp (thread principale)
              │  UIController:      disegno OpenCV, toast, barra di stato
              │  EnrollmentSession: macchina a stati della registrazione
              └─ tasti: Q esce, ESC annulla, INVIO conferma il nome
```

Regole da rispettare:
- `imshow` e `waitKey` vanno chiamati solo dal thread principale.
- YOLO e InsightFace vengono usati solo dal worker di inferenza.
- Il pool di connessioni PostgreSQL è thread-safe e viene usato da worker e thread principale. Tiene sempre aperte due connessioni, una per ciascun thread; una connessione caduta viene scartata e riaperta alla richiesta successiva.
- `RecognitionService` ha lo stato dei saluti protetto da lock, perché `register_greeting` viene chiamato dal thread principale.
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
4. `find_nearest`: `SELECT ... ORDER BY embedding <=> vettore LIMIT TOP_K_NEIGHBORS`. La similarità è `1 - distanza coseno`.
5. Se la migliore similarità è maggiore o uguale a `RECOGNITION_THRESHOLD`, la persona è riconosciuta. Altrimenti è sconosciuta.

## 6. Registrazione (EnrollmentSession)

Stati: `IDLE → NAME_INPUT → CAPTURING → COMPLETED | CANCELLED`.

- **Avvio (`consider_start`):** solo con un unico volto nel fotogramma, sconosciuto, largo almeno `MIN_FACE_WIDTH_PX`, e fuori dal periodo di snooze (`UNKNOWN_SNOOZE_SECONDS`, predefinito 60 secondi dopo un annullamento). La condizione deve valere per almeno 3 risultati di inferenza consecutivi e per `UNKNOWN_CONFIRM_SECONDS` (predefinito 1.5): un singolo fotogramma mosso o di profilo non fa comparire la richiesta del nome.
- **Nome:** almeno 2 caratteri, senza duplicati (confronto senza distinzione di maiuscole). Timeout di 60 secondi.
- **Campione valido (`offer_frame` / `offer_detection`):** un solo volto nel fotogramma, non riconosciuto come persona già registrata, `face_width ≥ MIN_FACE_WIDTH_PX` (90), nitidezza `≥ MIN_SHARPNESS` (60), similarità con il primo campione `≥ 0.50` (stessa persona), similarità con gli altri campioni `≤ 0.985` (varietà), intervallo minimo di 0.4 secondi tra due campioni. Timeout di 30 secondi per la cattura.
- **Salvataggio:** una sola transazione che inserisce la riga in `persons`, gli embedding con `source='enrollment'` e la riga `ENROLLED` in `access_logs`.
- **Dopo il salvataggio:** `FaceAccessApp` ignora i risultati di inferenza calcolati sui fotogrammi precedenti, che vedono ancora la persona come sconosciuta. Senza questo controllo la richiesta del nome ricomparirebbe subito.

## 7. Apprendimento continuo

Per ogni persona il servizio tiene in memoria un buffer di embedding con similarità `≥ CONTINUOUS_LEARNING_THRESHOLD` (0.55) e nitidezza sufficiente. Nei `CONTINUOUS_MIN_INTERVAL_SECONDS` (600 secondi) successivi a un aggiornamento il buffer non viene alimentato. Quando il buffer raggiunge `CONTINUOUS_BATCH_SIZE` (5):

1. calcola la media dei vettori;
2. la normalizza in L2;
3. la inserisce con `source='continuous'` e, come qualità, la nitidezza media del gruppo;
4. se i campioni `continuous` superano `MAX_CONTINUOUS_EMBEDDINGS` (20), elimina i più vecchi;
5. ricalcola `persons.sample_count` contando gli embedding rimasti.

## 8. Debouncing dei saluti

`RecognitionService._claim_greeting` usa un dizionario `{person_id: istante dell'ultimo saluto}` protetto da lock. Il saluto è concesso solo se è trascorso `GREETING_COOLDOWN_SECONDS`. `register_greeting()` evita il doppio saluto subito dopo una registrazione.

## 9. Schema del database (`sql/init.sql`)

- **persons**: `id SERIAL PK`, `full_name VARCHAR(120)`, `is_active`, `sample_count`, `created_at`, `last_access_at`. Indice unico su `LOWER(full_name)`.
- **face_embeddings**: `id BIGSERIAL PK`, `person_id FK → persons ON DELETE CASCADE`, `embedding VECTOR(512)`, `quality_score REAL`, `source ('enrollment' | 'continuous')`, `created_at`. Indice HNSW con `vector_cosine_ops`.
- **access_logs**: `id BIGSERIAL PK`, `person_id FK → persons ON DELETE SET NULL`, `event_type ('GREETED' | 'ENROLLED')`, `similarity REAL`, `occurred_at`.

Il database non contiene immagini, solo vettori e metadati.

## 10. Configurazione (`.env`, letto da `app/config.py`)

| Variabile | Predefinito | Significato |
|---|---|---|
| DB_HOST / DB_PORT / DB_NAME / DB_USER | 127.0.0.1 / 5432 / face_access / face_app | Connessione PostgreSQL. Non usare `localhost`: su Windows prova prima IPv6 e ogni connessione impiega 2 secondi |
| DB_PASSWORD | (obbligatoria) | Password, non può essere il valore di esempio |
| CAMERA_INDEX / CAMERA_WIDTH / CAMERA_HEIGHT | 0 / 1280 / 720 | Webcam |
| YOLO_MODEL_PATH | models/yolov8n-face.pt | Pesi YOLO per i volti (con fallback su yolov8n.pt) |
| YOLO_CONFIDENCE / YOLO_IMAGE_SIZE | 0.5 / 640 | Parametri di YOLO (la dimensione deve essere un multiplo di 32) |
| INSIGHTFACE_MODEL / INSIGHTFACE_DET_SIZE | buffalo_l / 320 | Modello InsightFace e risoluzione del rilevatore interno (multiplo di 32) |
| RECOGNITION_THRESHOLD | 0.40 | Soglia di riconoscimento (similarità coseno) |
| CONTINUOUS_LEARNING_THRESHOLD | 0.55 | Soglia per l'apprendimento continuo (deve essere ≥ della soglia di riconoscimento) |
| GREETING_COOLDOWN_SECONDS | 300 | Debouncing dei saluti |
| UNKNOWN_SNOOZE_SECONDS | 60 | Attesa prima di riproporre una registrazione annullata |
| UNKNOWN_CONFIRM_SECONDS | 1.5 | Tempo per cui un volto sconosciuto deve restare inquadrato prima della richiesta del nome |
| ENROLLMENT_SAMPLES | 8 | Campioni per la registrazione (5–10) |
| MIN_FACE_WIDTH_PX / MIN_SHARPNESS | 90 / 60 | Qualità minima dei campioni |
| CONTINUOUS_BATCH_SIZE / CONTINUOUS_MIN_INTERVAL_SECONDS | 5 / 600 | Parametri dell'apprendimento continuo |
| MAX_CONTINUOUS_EMBEDDINGS | 20 | Limite di embedding continuous per persona |
| TOP_K_NEIGHBORS | 5 | Numero di vicini interrogati in pgvector |

## 11. Mappa dei file

- `app/config.py`: `Settings` (dataclass immutabile), validazione, `ConfigurationError`
- `app/models.py`: `BBox`, `Match`, `FaceDetection`, `FrameResult`
- `app/camera_stream.py`: `CameraStream`, thread di cattura con riconnessione
- `app/face_processor.py`: `FaceProcessor`, pipeline YOLO + InsightFace, funzioni di ritaglio e nitidezza
- `app/database_manager.py`: `DatabaseManager`, pool, query, `DatabaseError`, `to_pgvector`
- `app/recognition_service.py`: `RecognitionService`, soglia, debouncing, apprendimento continuo
- `app/inference_worker.py`: `InferenceWorker`, thread di inferenza, risultati, coda dei saluti
- `app/enrollment.py`: `EnrollmentSession`, `EnrollmentState`, registrazione
- `app/ui_controller.py`: `UIController`, disegno OpenCV, toast, pannelli
- `app/main.py`: `FaceAccessApp` (composizione, ciclo UI, tasti) e `main()`
- `tests/`: test `unittest` della logica, con database, orologio e volti simulati in `tests/helpers.py`. Si eseguono con `python -m unittest`
- `sql/init.sql`, `docker-compose.yml`, `.env.example`, `.gitignore`, `requirements.txt`, `README.md`, `.vscode/launch.json`, `models/LEGGIMI.txt`

## 12. Convenzioni di codice

- PEP 8, type hints ovunque, docstring in italiano, identificatori in inglese.
- Logging con il modulo `logging`, mai `print`.
- Errori gestiti con eccezioni dedicate: `ConfigurationError` per la configurazione, `DatabaseError` per il database.
- Testi a schermo in OpenCV senza caratteri accentati, perché il font Hershey non li supporta.
- Credenziali solo nel file `.env`, mai nel codice.

## 13. Limiti noti

- Senza pesi `yolov8n-face.pt` si usa il rilevamento di persone: un volto viene trovato solo se YOLO riconosce la persona, e ogni persona in più nel fotogramma allunga l'inferenza (circa 0.2 secondi con una o due persone, 0.7 con sei).
- Un volto già registrato, visto con similarità sotto soglia (occhiali, angolazione estrema, scarsa luce), viene trattato come sconosciuto. Se resta sotto soglia per tutta la durata della conferma e della cattura, il sistema può ancora proporre e completare una registrazione duplicata; se durante la cattura torna sopra soglia, i campioni vengono scartati.
- I nomi si digitano solo con caratteri ASCII: niente lettere accentate, per il limite del font Hershey.
- La porta di PostgreSQL è pubblicata solo su `127.0.0.1` (`docker-compose.yml`).
- Il cooldown dei saluti è in memoria: al riavvio si azzera.
- Non c'è tracking tra fotogrammi. Con più volti vengono riconosciuti tutti, ma la registrazione parte solo con un volto unico.
- Su CPU con 1280x720 l'inferenza gira a pochi fps, mentre il video resta fluido.
- L'utente applicativo `face_app` è anche superutente del container: va bene in sviluppo, va ristretto in produzione.
- Stato delle verifiche (8 ottobre 2026, Windows 11, Python 3.11.9, CPU):
  - la logica è coperta dai test in `tests/`;
  - le query sono state provate su PostgreSQL 16 con pgvector 0.8, compresi riavvio del database e query bloccate;
  - YOLO (modello generico) e InsightFace sono stati provati sulle immagini di esempio delle due librerie;
  - l'applicazione completa (finestra, tasti, thread, registrazione, riconoscimento, saluto, apprendimento continuo, caduta e ripresa di sorgente video e database) è stata provata con una sorgente video simulata: circa 30 fps di video e 8 fps di inferenza;
  - **non ancora provati:** la cattura da una webcam fisica (sul PC di prova non era presente nessuna fotocamera) e i pesi `yolov8n-face.pt`. Le soglie `MIN_SHARPNESS` e `MIN_FACE_WIDTH_PX` vanno quindi tarate con la webcam reale.
- Aspetti GDPR e normativa sul lavoro: vedi la sezione 11 del README.

## 14. Possibili evoluzioni

- Tracking tra fotogrammi (ad esempio ByteTrack o IoU) per evitare identificazioni ripetute.
- GPU con `onnxruntime-gpu` e PyTorch con CUDA.
- Interfaccia Tkinter o PySide6 al posto di OpenCV, oppure API REST per integrazione con altri sistemi.
- Cifratura degli embedding a riposo.
- Ruoli utente, pannello di amministrazione e esportazione dei log.
- Test di integrazione con un database PostgreSQL di test: quelli attuali simulano il database.

## 15. Istruzioni per l'assistente

- Rispondi in italiano. Codice e commenti in italiano, identificatori in inglese.
- Prima di modificare un file, leggilo per intero e mantieni le responsabilità dei moduli.
- Non spostare la logica OpenCV nei thread di inferenza, e non eseguire modelli AI nel thread della UI.
- Non inserire credenziali nel codice: usa `.env`.
- Ogni modifica allo schema va riflessa in `sql/init.sql` e in `DatabaseManager`.
- Se aggiungi dipendenze, aggiornale in `requirements.txt` con la versione fissata e una riga di spiegazione.
- Chiedi conferma prima di modifiche che toccano dati biometrici, conservazione dei dati o soglie di sicurezza.
