# Face Access: riconoscimento biometrico in tempo reale

Applicazione desktop in Python che usa la webcam per riconoscere le persone registrate, salutarle per nome e registrare i nuovi volti. Usa **YOLOv8** per il rilevamento e **InsightFace (ArcFace)** per gli embedding biometrici. Il confronto avviene sulla telecamera, con una galleria in memoria; registrazioni, saluti e aggiornamenti dei profili vengono pubblicati via **MQTT** al server, che li salva nel database.

Questa è la repository della **telecamera**. Database, broker MQTT e chatbot sono nella repository **`face-access-server`**, che deve essere in esecuzione e raggiungibile in rete.

> **Attenzione privacy:** i dati biometrici sono una categoria particolare di dati personali (art. 9 GDPR). Prima di usare il sistema con altre persone leggi la sezione 11.

---

## 1. Requisiti

- Windows 10/11, macOS o Linux, con una webcam
- Python 3.11 (3.10 compatibile)
- Il server Face Access (repository `face-access-server`) in esecuzione sulla stessa rete locale
- Visual Studio Code con l'estensione **Python** (Microsoft)
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
│   ├── liveness.py             # Anti-spoofing: verifica che il volto non sia una foto (LivenessTracker)
│   ├── greeting_policy.py      # Regole di "Benvenuto" e "Bentornato" (GreetingPolicy)
│   ├── gallery.py              # Galleria locale delle persone registrate (Gallery)
│   ├── mqtt_link.py            # Collegamento MQTT con il server (MqttLink)
│   ├── person_store.py         # Letture sulla galleria, scritture via MQTT (PersonStore)
│   ├── contract.py             # Topic e formato dei messaggi MQTT
│   ├── new_face_selector.py    # Scelta del volto sconosciuto da registrare (NewFaceSelector)
│   ├── enrollment.py           # Registrazione dei nuovi volti (EnrollmentSession)
│   └── ui_controller.py        # Interfaccia grafica (UIController)
├── tests/                      # Test della logica (senza webcam, modelli e server)
├── models/                     # Qui va il file yolov8n-face.pt
├── .vscode/launch.json         # Configurazione di avvio per VS Code
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
2. Apri `.env` e imposta il collegamento al server:
   - `MQTT_HOST`: indirizzo IP del computer su cui gira il server (`127.0.0.1` se è lo stesso computer);
   - `MQTT_PORT`: lo stesso valore di `MQTT_PORT` nel `.env` del server;
   - `MQTT_PASSWORD`: lo stesso valore di `MQTT_CAMERA_PASSWORD` nel `.env` del server.
3. Non condividere e non committare mai il file `.env`.

## 4. Collegamento al server

Avvia prima il server (README di `face-access-server`: `docker compose up -d --build`). La telecamera non parla con il database: scambia messaggi con il broker MQTT del server.

```
 Telecamera ── eventi (registrazioni, saluti, embedding) ──► broker MQTT ──► ingest ──► PostgreSQL
 Telecamera ◄── galleria (persone ed embedding) ──────────── broker MQTT ◄── ingest
```

L'applicazione parte anche se il server non è raggiungibile e continua a ritentare la connessione. La barra di stato in basso mostra:

- `Galleria: OK` quando l'elenco delle persone è stato ricevuto, `ATTESA` prima;
- `Server: OK` quando il broker è raggiungibile e il servizio che salva gli eventi è attivo, `OFF` altrimenti.

Il traffico MQTT non è cifrato: usa il sistema solo su una rete locale fidata.

## 5. Come funziona la galleria locale

La galleria è la copia, in memoria, delle persone registrate. La fonte di verità resta il database del server.

- **Come si riempie.** Per ogni persona il server tiene sul broker un messaggio *retained* con nome, ultimo saluto e tutti gli embedding. Alla connessione la telecamera li riceve tutti, insieme all'elenco delle persone valide. Quando le ha ricevute tutte, la galleria è pronta.
- **Come riconosce.** A ogni volto calcola la similarità coseno con tutti gli embedding della galleria (un prodotto di matrici, meno di un millisecondo) e tiene il più simile. Se supera `RECOGNITION_THRESHOLD` la persona è riconosciuta.
- **Come si aggiorna.** Una persona appena registrata entra subito nella galleria locale, senza attendere il server. Saluti e apprendimento continuo vengono pubblicati; il server li salva e ripubblica la persona aggiornata. Le modifiche fatte a mano sul database (rinomina, disattivazione, cancellazione) arrivano entro 30 secondi.
- **Se il server non risponde.** Il riconoscimento continua con la galleria già ricevuta. Gli eventi restano in coda (fino a 1000) e partono alla riconnessione. Se l'applicazione viene chiusa prima, gli eventi ancora in coda vanno persi.
- **Prima che la galleria arrivi** i volti compaiono in grigio come "Non verificato" e non viene proposta nessuna registrazione: senza elenco, ogni persona sembrerebbe nuova.
- **Niente su disco.** Alla chiusura la galleria sparisce; al riavvio viene ricostruita dai messaggi del broker.

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
python -c "import cv2, ultralytics, insightface, paho.mqtt; print('OK - OpenCV', cv2.__version__)"
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
- **Verifica anti-spoofing:** prima di salutare o registrare qualcuno, il sistema controlla che il volto sia tridimensionale e non una foto. Basta muovere la testa di una decina di gradi, cosa che di solito accade da sola mentre ci si avvicina: finché la verifica non è riuscita il riquadro è color ambra con "verifica in corso" e, dopo due secondi, "gira leggermente la testa". Una foto stampata o mostrata su un telefono resta color ambra: non viene salutata, non genera accessi e non può essere registrata. Un video della persona riprodotto su uno schermo, invece, supera la verifica. Si disattiva con `LIVENESS_CHECK=0` in `.env`.
- **Persona nuova:** compare un riquadro rosso "Sconosciuto". Se il volto resta sconosciuto e abbastanza vicino alla webcam per circa un secondo e mezzo, il video si scurisce, al centro compare una finestra per inserire il nome e il suo riquadro diventa color ambra ("In registrazione"). Con più volti nell'inquadratura viene scelto quello sconosciuto più vicino; le persone già note non bloccano la registrazione. Digita il nome e premi INVIO. Poi resta davanti alla webcam: il sistema acquisisce da solo 8 campioni validi, muovendo leggermente la testa e variando un po' la luce. Se a campioni completi il volto non è ancora stato verificato, il pannello chiede di girare lentamente la testa a destra e a sinistra; senza verifica la registrazione si annulla dopo 30 secondi. Al termine compare "Benvenuto, Nome!", una sola volta. Con ESC annulli: il sistema non riproporrà registrazioni per 60 secondi.
- **Durante la cattura** il sistema segue solo il volto che ha fatto comparire la richiesta e ignora gli altri. Il pannello in basso spiega perché un campione viene scartato: volto troppo lontano, immagine sfocata (con il valore misurato e quello minimo), volto non più presente, oppure volto che appartiene a un profilo già registrato.
- **Richiesta ritirata in automatico:** se il volto in registrazione viene riconosciuto come persona già registrata per due risultati di fila, la registrazione si annulla per non creare un doppione. Se il volto si allontana per 10 secondi prima che sia stato digitato qualcosa, la richiesta di nome si chiude da sola.
- **Più messaggi insieme:** i messaggi in alto vengono impilati (fino a tre), così un saluto non ne copre un altro.
- **Galleria non ancora ricevuta:** la barra di stato mostra `Galleria: ATTESA` e i volti compaiono in grigio come "Non verificato". Il riconoscimento parte da solo quando il server consegna l'elenco delle persone.
- **Server non raggiungibile dopo l'avvio:** la barra di stato mostra `Server: OFF`. Riconoscimento e registrazione continuano; gli eventi vengono inviati alla riconnessione.
- **Apprendimento continuo:** i riconoscimenti molto sicuri arricchiscono il profilo con nuovi embedding mediati, al massimo ogni 10 minuti. Il nuovo embedding entra nella galleria quando il server lo ha salvato.

Tasti: `Q` esce · `INVIO` conferma il nome · `BACKSPACE` cancella un carattere · `ESC` annulla la registrazione.

## 11. Privacy e aspetti legali

Il sistema tratta dati biometrici, che il GDPR (art. 9) considera una categoria particolare di dati personali. Prima di usarlo con persone reali:

- definisci la base giuridica, di norma un consenso esplicito, informato e revocabile;
- informa gli interessati su finalità, tempi di conservazione e diritti;
- il codice non salva immagini del volto, ma solo vettori numerici; restano comunque dati biometrici. La verifica anti-spoofing tiene in memoria per pochi secondi un ritaglio del volto in scala di grigi, che non viene salvato né inviato;
- la telecamera tiene i vettori solo in memoria e li riceve e li invia sulla rete locale, senza cifratura;
- le persone si cancellano dal database del server (README di `face-access-server`): la telecamera le rimuove dalla propria galleria in un paio di secondi.

Se il sistema serve a rilevare presenze o a controllare dipendenti, in Italia si applicano vincoli molto stringenti (in particolare l'art. 4 dello Statuto dei Lavoratori e i provvedimenti del Garante per la protezione dei dati personali). Prima di attivarlo in questi contesti consulta un professionista.

## 12. Risoluzione dei problemi

- **`py -3.11` risponde "No suitable Python runtime found":** Python 3.11 non è installato (sezione 1). Con una versione più recente, ad esempio la 3.14, l'installazione delle librerie fallisce.
- **`Microsoft Visual C++ 14.0 or greater is required`** durante l'installazione di insightface: installa i Build Tools di Visual Studio con il carico di lavoro C++ (sezione 1).
- **`Unable to find a compatible Visual Studio installation`** durante l'installazione di insightface, con i Build Tools già installati: il terminale ha la variabile d'ambiente `NoDefaultCurrentDirectoryInExePath` (la impostano alcuni strumenti che aprono un terminale per conto tuo) e lo script di Visual Studio non trova `vswhere.exe`. Usa un normale terminale PowerShell, oppure esegui `Remove-Item Env:NoDefaultCurrentDirectoryInExePath` e ripeti i comandi della sezione 7 da `pip install -r requirements.txt` in poi.
- **`Galleria: ATTESA` e `Server: OFF` che non cambiano:** il broker non è raggiungibile. Controlla `MQTT_HOST` e `MQTT_PORT` in `.env`, che il server sia avviato e che il firewall del server lasci passare la porta (README di `face-access-server`, sezione 5).
- **Nel log `Connessione al broker MQTT rifiutata: Not authorized`:** `MQTT_PASSWORD` è diversa da `MQTT_CAMERA_PASSWORD` nel `.env` del server.
- **`Galleria: ATTESA` con il broker raggiungibile:** il servizio `ingest` del server non ha ancora pubblicato l'elenco delle persone. Controlla `docker compose logs ingest` sul server.
- **Una persona appena registrata sparisce dopo qualche secondo:** il server ha rifiutato la registrazione perché il nome appartiene già a un'altra persona (ad esempio una persona disattivata). Il motivo è nel log dell'ingest.
- **La cattura resta ferma su "Immagine sfocata":** il messaggio mostra la nitidezza misurata e quella minima. Migliora la luce, oppure abbassa `MIN_SHARPNESS` in `.env` fino a un valore di poco inferiore a quello misurato.
- **La richiesta del nome compare troppo presto o troppo tardi:** regola `UNKNOWN_CONFIRM_SECONDS` in `.env`.
- **Il riquadro resta color ambra con "gira leggermente la testa":** la verifica anti-spoofing non vede abbastanza movimento tra le parti del volto. Avvicinati, migliora la luce e gira lentamente la testa a destra e a sinistra. Se succede spesso anche girando la testa, abbassa `LIVENESS_MIN_PARALLAX` in `.env` (ad esempio da 0.07 a 0.05); se una foto viene accettata, alzalo.
- **"Bentornato" non compare:** è previsto se la persona è stata vista negli ultimi 10 minuti. Il tempo si regola con `WELCOME_BACK_ABSENCE_SECONDS` in `.env`. La vecchia variabile `GREETING_COOLDOWN_SECONDS` non viene più letta e si può cancellare.
- **Schermo nero o webcam non trovata:** cambia `CAMERA_INDEX` in `.env` (prova 0, 1, 2...) e riavvia l'app.
- **FPS bassi:** in `.env` riduci `YOLO_IMAGE_SIZE` a 320, `INSIGHTFACE_DET_SIZE` a 256, oppure `CAMERA_WIDTH` e `CAMERA_HEIGHT` a 640 e 480.
- **Riconoscimento poco affidabile:** migliora la luce, oppure regola `RECOGNITION_THRESHOLD` (valori tipici tra 0.35 e 0.50).
- **Errore di conflitto su `cv2`:** ripeti i due comandi finali della sezione 7.

## 13. Test della logica

La cartella `tests/` verifica le parti che non dipendono da webcam e modelli: registrazione, soglia di riconoscimento, anti-spoofing, saluti, apprendimento continuo, configurazione, galleria locale e formato degli eventi MQTT (con il collegamento al server simulato). Usa `unittest`, già incluso in Python. Con `(.venv)` attivo, dalla cartella del progetto:

```bash
python -m unittest
```

I test non sostituiscono la prova con webcam e server reali: non eseguono YOLO e InsightFace e non si collegano al broker.
