# Face Access: riconoscimento biometrico in tempo reale

Applicazione desktop in Python che usa la webcam per riconoscere le persone registrate, salutarle per nome e registrare i nuovi volti. Usa **YOLOv8** per il rilevamento, **InsightFace (ArcFace)** per gli embedding biometrici e **PostgreSQL + pgvector** per il confronto vettoriale.

> **Attenzione privacy:** i dati biometrici sono una categoria particolare di dati personali (art. 9 GDPR). Prima di usare il sistema con altre persone leggi la sezione 11.

---

## 1. Requisiti

- Windows 10/11, macOS o Linux, con una webcam
- Python 3.11 (3.10 compatibile)
- Docker Desktop
- Visual Studio Code con l'estensione **Python** (Microsoft)
- DBeaver Community (consigliato per ispezionare il database)
- Solo Windows: **Build Tools per Visual Studio** con il carico di lavoro "Sviluppo di applicazioni desktop con C++". Serve a compilare `insightface`.

Su Windows puoi installare Python 3.11 e i Build Tools da PowerShell:

```powershell
winget install Python.Python.3.11
winget install Microsoft.VisualStudio.2022.BuildTools --override "--passive --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

Con `py -0p` vedi le versioni di Python installate: deve comparire la 3.11 (o la 3.10). Le versioni dalla 3.13 in poi non sono compatibili con le librerie fissate in `requirements.txt`.

## 2. Struttura del progetto

```
allenamento riconoscimento facciale AI/
├── app/
│   ├── __init__.py
│   ├── main.py                 # Avvio applicazione, ciclo UI, tasti
│   ├── config.py               # Lettura e validazione del file .env
│   ├── models.py               # Strutture dati condivise
│   ├── camera_stream.py        # Thread di cattura webcam (CameraStream)
│   ├── face_processor.py       # YOLO + InsightFace (FaceProcessor)
│   ├── inference_worker.py     # Thread di inferenza
│   ├── recognition_service.py  # Soglie, saluti, apprendimento continuo
│   ├── greeting_policy.py      # Regole di "Benvenuto" e "Bentornato" (GreetingPolicy)
│   ├── database_manager.py     # PostgreSQL + pgvector (DatabaseManager)
│   ├── new_face_selector.py    # Scelta del volto sconosciuto da registrare (NewFaceSelector)
│   ├── enrollment.py           # Registrazione dei nuovi volti (EnrollmentSession)
│   └── ui_controller.py        # Interfaccia OpenCV (UIController)
├── sql/init.sql                # Schema del database
├── tests/                      # Test della logica (senza webcam, modelli e database)
├── models/                     # Qui va il file yolov8n-face.pt
├── .vscode/launch.json         # Configurazione di avvio per VS Code
├── docker-compose.yml          # PostgreSQL 16 + pgvector
├── requirements.txt            # Librerie Python con versioni e spiegazioni
├── .env.example                # Modello del file di configurazione
├── .gitignore                  # Esclude .env, .venv e i pesi dei modelli
├── README.md                   # Questa guida
└── CONTESTO_PROGETTO.md        # Documento di contesto per altre chat
```

## 3. Configurazione del file .env

1. Copia il file di esempio:
   - Windows (PowerShell): `Copy-Item .env.example .env`
   - macOS/Linux: `cp .env.example .env`
2. Apri `.env` e imposta `DB_PASSWORD` con una password personale. Usa solo lettere, cifre e i simboli `-`, `_` e `.`: Docker e Python leggono in modo diverso i caratteri speciali di questo file.
3. Non condividere e non committare mai il file `.env`.

La password viene applicata al database solo alla prima creazione del volume. Se la cambi in seguito, ricrea il database con `docker compose down -v` (cancella tutti i dati).

## 4. Avvio di PostgreSQL con Docker

Dalla cartella del progetto:

```bash
docker compose up -d
docker compose ps
```

Il servizio `postgres` deve risultare `healthy` (attendi qualche secondo). Lo script `sql/init.sql` viene eseguito automaticamente solo la prima volta che il volume dati viene creato. Per verificare che le tabelle esistano:

```bash
docker compose exec postgres psql -U face_app -d face_access -c "\dt"
```

Devi vedere `persons`, `face_embeddings` e `access_logs`.

La porta del database è pubblicata solo su `127.0.0.1`: è raggiungibile da questo computer (applicazione e DBeaver), non dagli altri dispositivi della rete. Per questo in `.env` `DB_HOST` vale `127.0.0.1` e non `localhost`: su Windows `localhost` prova prima l'indirizzo IPv6 e ogni nuova connessione impiegherebbe circa 2 secondi.

## 5. Connessione con DBeaver

1. Apri DBeaver e scegli **Nuova connessione**, poi **PostgreSQL**, poi **Avanti**.
2. Compila i campi:
   - Host: `127.0.0.1`
   - Porta: quella di `DB_PORT` nel file `.env` (`5432` se non l'hai cambiata)
   - Database: `face_access`
   - Utente: `face_app`
   - Password: quella impostata nel file `.env`
3. Al primo collegamento DBeaver propone di scaricare il driver: accetta con **Download**.
4. Clicca **Test connessione**, poi **Fine**.
5. Per verificare pgvector apri un editor SQL (Ctrl+]) ed esegui `SELECT extname FROM pg_extension;`. Nell'elenco deve comparire `vector`.
6. Se non vedi le tabelle, premi F5 sul nodo `face_access` per aggiornare la vista.

## 6. Ambiente virtuale in VS Code

1. Apri VS Code e scegli **File → Apri cartella...**, poi seleziona la cartella del progetto (`allenamento riconoscimento facciale AI`).
2. Apri il terminale integrato (Ctrl+`) e crea l'ambiente virtuale:
   - Windows: `py -3.11 -m venv .venv` seguito da `.venv\Scripts\activate`
   - macOS/Linux: `python3.11 -m venv .venv` seguito da `source .venv/bin/activate`
3. Premi Ctrl+Shift+P, scegli **Python: Select Interpreter** e seleziona l'interprete dentro `.venv`.
4. Se PowerShell blocca l'attivazione con un errore di "execution policy", esegui una sola volta:
   `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

Quando l'ambiente è attivo, il prompt del terminale mostra `(.venv)`.

## 7. Installazione delle dipendenze

Con `(.venv)` attivo esegui questi comandi, nell'ordine:

```bash
python -m pip install --upgrade pip
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip uninstall -y opencv-python opencv-python-headless
pip install opencv-python==4.10.0.84
```

Il penultimo e l'ultimo comando risolvono un conflitto noto: `insightface` installa anche `opencv-python-headless`, che entra in conflitto con `opencv-python`.

Verifica finale:

```bash
python -c "import cv2, ultralytics, insightface, psycopg2; print('OK - OpenCV', cv2.__version__)"
```

Se hai una scheda NVIDIA e vuoi usare la GPU, installa PyTorch con CUDA dal sito pytorch.org e sostituisci `onnxruntime` con `onnxruntime-gpu`. Per iniziare la versione CPU è la più semplice.

## 8. Modello YOLO per il rilevamento dei volti

Ultralytics fornisce modelli generici (persone, oggetti), non specifici per i volti. Scarica un file di pesi `yolov8n-face.pt` da una repository della comunità (cerca "yolov8 face" su GitHub o Hugging Face, verificando la licenza) e copialo in:

```
models/yolov8n-face.pt
```

Se il file manca, l'app usa `yolov8n.pt` (scaricato automaticamente da Ultralytics nella cartella `models/`) che rileva le persone, e cerca i volti nel ritaglio di ciascuna. Funziona, ma un volto viene trovato solo se YOLO riconosce la persona, e con molte persone nel fotogramma l'inferenza rallenta. Nel log compare un avviso.

## 9. Avvio dell'applicazione

- Dal terminale, nella cartella del progetto: `python -m app.main`
- Da VS Code: premi F5 (configurazione "Face Access" in `.vscode/launch.json`)

Al primo avvio InsightFace scarica il pacchetto di modelli `buffalo_l` (circa 280 MB) in `~/.insightface/models`. Serve connessione internet solo questa volta.

## 10. Come si usa

- **Persona già registrata:** compare un riquadro verde con nome e similarità. Il saluto "Bentornato, Nome!" compare solo quando la persona rientra nell'inquadratura dopo esserne rimasta fuori per almeno 10 minuti (`WELCOME_BACK_ABSENCE_SECONDS`), contati dall'ultimo istante in cui è stata vista. Chi resta davanti alla webcam non viene salutato di nuovo.
- **Persona nuova:** compare un riquadro rosso "Sconosciuto". Se il volto resta sconosciuto e abbastanza vicino alla webcam per circa un secondo e mezzo, compare la richiesta di nome e il suo riquadro diventa arancione ("In registrazione"). Con più volti nell'inquadratura viene scelto quello sconosciuto più vicino; le persone già note non bloccano la registrazione. Digita il nome e premi INVIO. Poi resta davanti alla webcam: il sistema acquisisce da solo 8 campioni validi, muovendo leggermente la testa e variando un po' la luce. Al termine compare "Benvenuto, Nome!", una sola volta. Con ESC annulli: il sistema non riproporrà registrazioni per 60 secondi.
- **Durante la cattura** il sistema segue solo il volto che ha fatto comparire la richiesta e ignora gli altri. Il pannello in basso spiega perché un campione viene scartato: volto troppo lontano, immagine sfocata (con il valore misurato e quello minimo), volto non più presente, oppure volto che appartiene a un profilo già registrato.
- **Richiesta ritirata in automatico:** se il volto in registrazione viene riconosciuto come persona già presente nel database per due risultati di fila, la registrazione si annulla per non creare un doppione. Se il volto si allontana per 10 secondi prima che sia stato digitato qualcosa, la richiesta di nome si chiude da sola.
- **Più messaggi insieme:** i messaggi in alto vengono impilati (fino a tre), così un saluto non ne copre un altro.
- **Database non raggiungibile:** la barra di stato mostra `DB: ERRORE` e i volti compaiono in grigio come "Non verificato". Il riconoscimento riprende da solo quando il database torna disponibile.
- **Apprendimento continuo:** i riconoscimenti molto sicuri arricchiscono il profilo con nuovi embedding mediati, al massimo ogni 10 minuti.

Tasti: `Q` esce · `INVIO` conferma il nome · `BACKSPACE` cancella un carattere · `ESC` annulla la registrazione.

## 11. Privacy e aspetti legali

Il sistema tratta dati biometrici, che il GDPR (art. 9) considera una categoria particolare di dati personali. Prima di usarlo con persone reali:

- definisci la base giuridica, di norma un consenso esplicito, informato e revocabile;
- informa gli interessati su finalità, tempi di conservazione e diritti;
- il codice non salva immagini del volto, ma solo vettori numerici; restano comunque dati biometrici;
- per cancellare una persona basta eliminare la sua riga in `persons`: gli embedding si rimuovono in cascata, mentre i log restano anonimi (`person_id` diventa NULL).

Se il sistema serve a rilevare presenze o a controllare dipendenti, in Italia si applicano vincoli molto stringenti (in particolare l'art. 4 dello Statuto dei Lavoratori e i provvedimenti del Garante per la protezione dei dati personali). Prima di attivarlo in questi contesti consulta un professionista.

## 12. Risoluzione dei problemi

- **`py -3.11` risponde "No suitable Python runtime found":** Python 3.11 non è installato (sezione 1). Con una versione più recente, ad esempio la 3.14, l'installazione delle librerie fallisce.
- **`Microsoft Visual C++ 14.0 or greater is required`** durante l'installazione di insightface: installa i Build Tools di Visual Studio con il carico di lavoro C++ (sezione 1).
- **`Unable to find a compatible Visual Studio installation`** durante l'installazione di insightface, con i Build Tools già installati: il terminale ha la variabile d'ambiente `NoDefaultCurrentDirectoryInExePath` (la impostano alcuni strumenti che aprono un terminale per conto tuo) e lo script di Visual Studio non trova `vswhere.exe`. Usa un normale terminale PowerShell, oppure esegui `Remove-Item Env:NoDefaultCurrentDirectoryInExePath` e ripeti i comandi della sezione 7 da `pip install -r requirements.txt` in poi.
- **`docker compose up -d` non trova il motore Docker** (`failed to connect to the docker API`): avvia Docker Desktop e attendi che sia pronto.
- **`password authentication failed`:** la password in `.env` è diversa da quella con cui è stato creato il database, oppure contiene caratteri speciali (sezione 3).
- **La cattura resta ferma su "Immagine sfocata":** il messaggio mostra la nitidezza misurata e quella minima. Migliora la luce, oppure abbassa `MIN_SHARPNESS` in `.env` fino a un valore di poco inferiore a quello misurato.
- **La richiesta del nome compare troppo presto o troppo tardi:** regola `UNKNOWN_CONFIRM_SECONDS` in `.env`.
- **"Bentornato" non compare:** è previsto se la persona è stata vista negli ultimi 10 minuti. Il tempo si regola con `WELCOME_BACK_ABSENCE_SECONDS` in `.env`. La vecchia variabile `GREETING_COOLDOWN_SECONDS` non viene più letta e si può cancellare.
- **Schermo nero o webcam non trovata:** cambia `CAMERA_INDEX` in `.env` (prova 0, 1, 2...) e riavvia l'app.
- **`Connessione a PostgreSQL non riuscita`:** verifica con `docker compose ps`. Se la porta 5432 è occupata da un altro PostgreSQL, cambia `DB_PORT` in `.env` e rilancia `docker compose up -d`.
- **`Estensione pgvector non attiva` o tabella mancante:** ricrea il database con `docker compose down -v` seguito da `docker compose up -d`. Attenzione: il comando cancella tutti i dati.
- **FPS bassi:** in `.env` riduci `YOLO_IMAGE_SIZE` a 320, `INSIGHTFACE_DET_SIZE` a 256, oppure `CAMERA_WIDTH` e `CAMERA_HEIGHT` a 640 e 480.
- **Riconoscimento poco affidabile:** migliora la luce, oppure regola `RECOGNITION_THRESHOLD` (valori tipici tra 0.35 e 0.50).
- **Errore di conflitto su `cv2`:** ripeti i due comandi finali della sezione 7.

## 13. Test della logica

La cartella `tests/` verifica le parti che non dipendono da webcam e modelli: registrazione, soglia di riconoscimento, saluti, apprendimento continuo, configurazione e gestione degli errori del database (simulato). Usa `unittest`, già incluso in Python. Con `(.venv)` attivo, dalla cartella del progetto:

```bash
python -m unittest
```

I test non sostituiscono la prova con webcam e database reali: non eseguono YOLO, InsightFace né le query su PostgreSQL.
