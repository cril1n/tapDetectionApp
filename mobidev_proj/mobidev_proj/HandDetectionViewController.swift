import UIKit
import MediaPipeTasksVision
import AVFoundation
import CoreML

// MARK: - Struttura Dati per la Registrazione
/// Definisce la struttura di un singolo punto dati di movimento catturato.
/// Questi dati vengono raccolti in finestre per l'addestramento del modello.
struct MotionDataPoint: Codable {
    var relativeYVelocity: Float      // Velocità relativa del dito indice rispetto al polso sull'asse Y.
    var relativeYAcceleration: Float  // Accelerazione relativa del dito indice.
    var palmStabilityScore: Float     // Punteggio che indica la stabilità del palmo.
}

// MARK: - Controller Principale
class HandDetectionViewController: UIViewController {
    
    // MARK: - Modalità Operativa
    /// Enum per definire se l'app è in modalità di registrazione dati o di inferenza (utilizzo del modello).
    enum Mode {
        case recording // Modalità per raccogliere dati di allenamento.
        case inference // Modalità per eseguire predizioni con il modello addestrato.
    }
    
    // MARK: - Proprietà UI
    private let cameraPreviewView = UIView() // Vista che contiene il feed della fotocamera.
    private let statusLabel = UILabel() // Etichetta per mostrare lo stato attuale (es. "Registrando...", "Tap!").
    private let progressView = UIProgressView(progressViewStyle: .bar) // Barra di avanzamento per la registrazione.
    private let handSkeletonView = HandSkeletonView() // Vista custom per disegnare lo scheletro della mano.
    private let modeSelector = UISegmentedControl(items: ["Recording", "Inference"]) // Selettore per cambiare modalità.
    private var torchButton: UIButton! // Pulsante per attivare/disattivare la torcia.
    
    // MARK: - Proprietà MediaPipe
    private var handLandmarker: HandLandmarker? // Oggetto di MediaPipe per il rilevamento dei landmark della mano.
    private var captureSession: AVCaptureSession? // Gestisce il flusso di input/output della fotocamera.
    private var videoDataOutput: AVCaptureVideoDataOutput? // Fornisce i frame video per l'analisi.
    private var previewLayer: AVCaptureVideoPreviewLayer? // Layer per visualizzare il feed video.
    private let videoDataOutputQueue = DispatchQueue(label: "VideoDataOutput", qos: .userInitiated, attributes: [], autoreleaseFrequency: .workItem)

    // MARK: - Proprietà di Stato
    private var lastProcessedLandmarks: [NormalizedLandmark]? // Landmark della mano dell'ultimo frame processato.
    private var lastRelativeYVelocity: Float? // Velocità Y relativa dell'ultimo frame, usata per calcolare l'accelerazione.
    private var isTorchOn = false // Stato della torcia (accesa/spenta).

    /// Modalità corrente dell'app. Il `didSet` aggiorna l'interfaccia utente quando la modalità cambia.
    private var currentMode: Mode = .recording {
        didSet {
            updateUIVisibility()
        }
    }
    
    // MARK: - Proprietà per la Registrazione
    private var isRecordingWindow = false // Flag che indica se è in corso la registrazione di una finestra di dati.
    private var currentRecordingLabel: String = "sfondo" // Etichetta per la finestra corrente ("tap" o "sfondo").
    private var windowBuffer: [MotionDataPoint] = [] // Buffer che accumula i dati di movimento per una finestra.
    private let recordingWindowSize = 25 // Numero di frame da catturare per ogni finestra di registrazione.
    
    // MARK: - Proprietà per l'Infernza
    private var tapDetectionModel: TapDetector? // Il modello Core ML addestrato per il rilevamento del tap.
    private var isActionOnCooldown = false // Previene rilevamenti multipli in rapida successione (es. dopo un tap).

    // MARK: - Buffer per le Feature di Infernza
    /// Buffer che mantengono gli ultimi `predictionWindowSize` valori per le feature usate dal modello.
    private var velocityBuffer: [Double] = []
    private var accelerationBuffer: [Double] = []
    private var palmStabilityScoreBuffer: [Double] = []
    private let predictionWindowSize = 25 // Dimensione della finestra richiesta dal modello Core ML per la predizione.
    
    /// Callback eseguita quando il controller viene chiuso.
    var onDismiss: (() -> Void)?
    
    // MARK: - Ciclo di Vita del ViewController
    override func viewDidLoad() {
        super.viewDidLoad()
        setupUI() // Configura gli elementi dell'interfaccia utente.
        setupHandLandmarker() // Inizializza il rilevatore di mano di MediaPipe.
        setupCamera() // Configura la sessione di cattura video.
        loadModel() // Carica il modello Core ML.
    }
    
    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        startCamera() // Avvia la fotocamera quando la vista appare.
    }
    
    override func viewDidDisappear(_ animated: Bool) {
        super.viewDidDisappear(animated)
        stopCamera() // Ferma la fotocamera per risparmiare risorse.
        if isTorchOn {
            toggleTorch(forceOff: true) // Spegne la torcia se era accesa.
        }
    }
    
    /// Assicura che il layer di anteprima e lo scheletro si ridimensionino correttamente con la vista.
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        previewLayer?.frame = cameraPreviewView.bounds
        handSkeletonView.frame = cameraPreviewView.bounds
    }
    
    // MARK: - Configurazione UI e Fotocamera
    
    /// Imposta e posiziona tutti gli elementi dell'interfaccia utente.
    private func setupUI() {
        view.backgroundColor = .black
        
        // Impostazione delle viste principali (anteprima fotocamera e scheletro)
        cameraPreviewView.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(cameraPreviewView)
        
        handSkeletonView.translatesAutoresizingMaskIntoConstraints = false
        handSkeletonView.backgroundColor = .clear
        view.addSubview(handSkeletonView)
        
        // Impostazione degli elementi di stato e controllo
        statusLabel.translatesAutoresizingMaskIntoConstraints = false
        statusLabel.textAlignment = .center
        statusLabel.backgroundColor = UIColor.black.withAlphaComponent(0.7)
        statusLabel.textColor = .white
        statusLabel.layer.cornerRadius = 8
        statusLabel.clipsToBounds = true
        view.addSubview(statusLabel)
        
        progressView.translatesAutoresizingMaskIntoConstraints = false
        progressView.progress = 0.0
        progressView.trackTintColor = .systemGray
        progressView.progressTintColor = .systemBlue
        view.addSubview(progressView)
        
        modeSelector.translatesAutoresizingMaskIntoConstraints = false
        modeSelector.selectedSegmentIndex = 0
        modeSelector.backgroundColor = .darkGray
        modeSelector.addTarget(self, action: #selector(modeChanged), for: .valueChanged)
        view.addSubview(modeSelector)
        
        // Applicazione dei constraints AutoLayout
        NSLayoutConstraint.activate([
            modeSelector.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 10),
            modeSelector.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 20),
            modeSelector.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -20),
            
            progressView.topAnchor.constraint(equalTo: modeSelector.bottomAnchor, constant: 10),
            progressView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            progressView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            
            cameraPreviewView.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor),
            cameraPreviewView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            cameraPreviewView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            cameraPreviewView.bottomAnchor.constraint(equalTo: view.bottomAnchor),
            
            handSkeletonView.topAnchor.constraint(equalTo: cameraPreviewView.topAnchor),
            handSkeletonView.leadingAnchor.constraint(equalTo: cameraPreviewView.leadingAnchor),
            handSkeletonView.trailingAnchor.constraint(equalTo: cameraPreviewView.trailingAnchor),
            handSkeletonView.bottomAnchor.constraint(equalTo: cameraPreviewView.bottomAnchor),
            
            statusLabel.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 20),
            statusLabel.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -20),
            statusLabel.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -20),
            statusLabel.heightAnchor.constraint(equalToConstant: 44)
        ])
        
        // Setup dei pulsanti ausiliari
        setupCloseButton()
        setupTorchButton()
        setupRecordingControls()
        updateUIVisibility() // Imposta la visibilità iniziale degli elementi.
    }
    
    /// Configura il task HandLandmarker di MediaPipe.
    private func setupHandLandmarker() {
        // Cerca il modello `.task` nel bundle dell'app.
        guard let modelPath = Bundle.main.path(forResource: "hand_landmarker", ofType: "task") else {
            showAlert(title: "Error", message: "Model file not found.")
            return
        }
        
        // Imposta le opzioni per il rilevatore.
        let options = HandLandmarkerOptions()
        options.baseOptions.modelAssetPath = modelPath
        options.runningMode = .liveStream // Modalità per l'analisi di un flusso video in tempo reale.
        options.numHands = 1 // Rileva al massimo una mano.
        options.minHandDetectionConfidence = 0.5 // Confidenza minima per il rilevamento iniziale.
        options.minHandPresenceConfidence = 0.5 // Confidenza minima per confermare la presenza della mano nei frame successivi.
        options.minTrackingConfidence = 0.5 // Confidenza minima per il tracciamento della mano.
        options.handLandmarkerLiveStreamDelegate = self // Assegna il delegate per ricevere i risultati.
        
        do {
            // Crea l'istanza di HandLandmarker.
            handLandmarker = try HandLandmarker(options: options)
            updateStatus("✅ HandLandmarker configured")
        } catch {
            showAlert(title: "Error", message: "Could not initialize HandLandmarker: \(error.localizedDescription)")
        }
    }

    // MARK: - Controllo Fotocamera
    
    /// Inizializza e configura la sessione di cattura video (AVCaptureSession).
    private func setupCamera() {
        captureSession = AVCaptureSession()
        guard let captureSession = captureSession else { return }
        
        captureSession.beginConfiguration()
        captureSession.sessionPreset = .medium // Qualità media per bilanciare performance e dettaglio.
        
        // Imposta la fotocamera posteriore come input.
        guard let backCamera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            updateStatus("❌ Fotocamera non disponibile")
            return
        }
        
        print("✅ FPS ATTUALI: \(backCamera.activeVideoMinFrameDuration.timescale)")
        
        do {
            let input = try AVCaptureDeviceInput(device: backCamera)
            if captureSession.canAddInput(input) {
                captureSession.addInput(input)
            }
        } catch {
            updateStatus("❌ Errore input fotocamera")
            return
        }
        
        // Configura l'output dei dati video.
        videoDataOutput = AVCaptureVideoDataOutput()
        videoDataOutput?.setSampleBufferDelegate(self, queue: videoDataOutputQueue) // Imposta il delegate per ricevere i buffer dei frame.
        videoDataOutput?.alwaysDiscardsLateVideoFrames = true // Scarta i frame in ritardo per mantenere la fluidità.
        videoDataOutput?.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)] // Formato pixel.
        
        if let videoDataOutput = videoDataOutput, captureSession.canAddOutput(videoDataOutput) {
            captureSession.addOutput(videoDataOutput)
        }
        
        captureSession.commitConfiguration()
        setupPreviewLayer() // Imposta il layer di anteprima.
    }
    
    /// Crea e aggiunge il layer di anteprima video alla vista.
    private func setupPreviewLayer() {
        guard let captureSession = captureSession else { return }
        
        previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer?.frame = cameraPreviewView.bounds
        previewLayer?.videoGravity = .resizeAspectFill
        
        if let previewLayer = previewLayer {
            // Rimuove eventuali layer precedenti prima di aggiungerne uno nuovo.
            cameraPreviewView.layer.sublayers?.forEach { $0.removeFromSuperlayer() }
            cameraPreviewView.layer.addSublayer(previewLayer)
        }
    }
    
    /// Avvia la sessione di cattura su una coda in background.
    private func startCamera() {
        guard let captureSession = captureSession, !captureSession.isRunning else { return }
        videoDataOutputQueue.async {
            captureSession.startRunning()
        }
    }
    
    /// Ferma la sessione di cattura.
    private func stopCamera() {
        guard let captureSession = captureSession, captureSession.isRunning else { return }
        captureSession.stopRunning()
    }
    
    /// Carica il modello di classificazione Core ML.
    private func loadModel() {
            do {
                let config = MLModelConfiguration()
                tapDetectionModel = try TapDetector(configuration: config)
                print("✅ Modello 'TapDetector' caricato con successo.")
            } catch {
                print("❌ Errore durante il caricamento del modello Core ML: \(error)")
                showAlert(title: "Errore Modello", message: "Impossibile caricare il classificatore.")
            }
        }
    
    // MARK: - Processamento dei Risultati
    
    /// Funzione centrale che riceve i risultati da MediaPipe e li indirizza alla modalità corretta.
    private func processHandResults(_ result: HandLandmarkerResult) {
        // Se non viene rilevata una mano con almeno 18 landmark, resetta lo stato.
        guard let landmarks = result.landmarks.first, landmarks.count > 17 else {
                self.lastProcessedLandmarks = nil
                self.lastRelativeYVelocity = nil
                self.handSkeletonView.updatePoints([]) // Pulisce lo scheletro dalla UI.
                updateStatus(currentMode == .inference ? "In attesa della mano." : "👋 Pronto a registrare.")
                return
            }
        
        // Converte le coordinate normalizzate dei landmark in punti della vista.
        if let previewLayer = self.previewLayer {
            let convertedPoints = landmarks.map { landmark in
                let point = CGPoint(x: CGFloat(landmark.x), y: CGFloat(landmark.y))
                return previewLayer.layerPointConverted(fromCaptureDevicePoint: point)
            }
            self.handSkeletonView.updatePoints(convertedPoints) // Aggiorna il disegno dello scheletro.
        }
        
        // Inizializza le feature a zero.
        var relativeIndexYVelocity: Float = 0.0
        var relativeIndexYAcceleration: Float = 0.0
        var palmStabilityScore: Float = 0.0
        
        // Calcola le feature se sono disponibili i dati del frame precedente.
        if let lastLandmarks = self.lastProcessedLandmarks, let lastVel = self.lastRelativeYVelocity {
                // Estrae i landmark di interesse.
                let wrist = landmarks[0], lastWrist = lastLandmarks[0]
                let indexTip = landmarks[8], lastIndexTip = lastLandmarks[8]
                let middleMcp = landmarks[9], lastMiddleMcp = lastLandmarks[9]
                let ringMcp = landmarks[13], lastRingMcp = lastLandmarks[13]
                let pinkyMcp = landmarks[17], lastPinkyMcp = lastLandmarks[17]

                // Calcola la velocità relativa della punta del dito indice rispetto al polso.
                relativeIndexYVelocity = (indexTip.y - lastIndexTip.y) - (wrist.y - lastWrist.y)
                // Calcola l'accelerazione come differenza tra la velocità attuale e quella precedente.
                relativeIndexYAcceleration = relativeIndexYVelocity - lastVel
                
                // Calcola la velocità dei punti chiave del palmo.
                let wristVelocity = abs(wrist.y - lastWrist.y)
                let middleMcpVelocity = abs(middleMcp.y - lastMiddleMcp.y)
                let ringMcpVelocity = abs(ringMcp.y - lastRingMcp.y)
                let pinkyMcpVelocity = abs(pinkyMcp.y - lastPinkyMcp.y)
                
                // Il punteggio di stabilità è la somma delle velocità: un valore basso indica un palmo fermo.
                palmStabilityScore = wristVelocity + middleMcpVelocity + ringMcpVelocity + pinkyMcpVelocity
            }
        
        // Raggruppa le feature calcolate.
        let features = (
                velocityY: relativeIndexYVelocity,
                accelerationY: relativeIndexYAcceleration,
                stability: palmStabilityScore
            )
        
        // Inoltra le feature al gestore della modalità corrente.
        switch currentMode {
        case .recording:
            handleRecording(features: features)
        case .inference:
            handleInference(features: features)
        }
        
        // Salva lo stato corrente per il prossimo frame.
        self.lastProcessedLandmarks = landmarks
        self.lastRelativeYVelocity = relativeIndexYVelocity
    }
    
    // MARK: - Gestori Specifici per Modalità
    
    /// Gestisce la logica quando l'app è in modalità "Recording".
    private func handleRecording(features: (velocityY: Float, accelerationY: Float, stability: Float)) {
            if isRecordingWindow {
                // Crea un nuovo punto dati e lo aggiunge al buffer della finestra.
                let dataPoint = MotionDataPoint(
                    relativeYVelocity: features.velocityY,
                    relativeYAcceleration: features.accelerationY,
                    palmStabilityScore: features.stability
                )
                windowBuffer.append(dataPoint)
                
                // Aggiorna la barra di avanzamento.
                let progress = Float(windowBuffer.count) / Float(recordingWindowSize)
                progressView.setProgress(progress, animated: true)
                
                // Se la finestra è piena, la salva su file.
                if windowBuffer.count >= recordingWindowSize {
                    saveWindowToFile()
                    isRecordingWindow = false // Termina la registrazione della finestra.
                }
            }
        }
    
    /// Gestisce la logica quando l'app è in modalità "Inference".
    private func handleInference(features: (velocityY: Float, accelerationY: Float, stability: Float)) {
           guard let model = tapDetectionModel else {
               updateStatus("Modello non pronto")
               return
           }

            // Aggiunge le nuove feature ai buffer, mantenendo la dimensione della finestra.
            velocityBuffer.append(Double(features.velocityY))
            accelerationBuffer.append(Double(features.accelerationY))
            palmStabilityScoreBuffer.append(Double(features.stability))

           if velocityBuffer.count > predictionWindowSize {
               velocityBuffer.removeFirst()
               accelerationBuffer.removeFirst()
               palmStabilityScoreBuffer.removeFirst()
           }

           // Procede con la predizione solo quando i buffer sono pieni.
           guard velocityBuffer.count == predictionWindowSize else {
               let status = "Buffer... (\(velocityBuffer.count)/\(predictionWindowSize))"
               DispatchQueue.main.async { self.statusLabel.text = status }
               return
           }

           do {
               // Prepara l'input per il modello Core ML (un MLMultiArray).
               let modelInputArray = try MLMultiArray(shape: [1, 3, NSNumber(value: predictionWindowSize)], dataType: .double)
               for i in 0..<predictionWindowSize {
                   modelInputArray[[0, 0, i] as [NSNumber]] = NSNumber(value: velocityBuffer[i])
                   modelInputArray[[0, 1, i] as [NSNumber]] = NSNumber(value: accelerationBuffer[i])
                   modelInputArray[[0, 2, i] as [NSNumber]] = NSNumber(value: palmStabilityScoreBuffer[i])
               }

               // Esegue la predizione.
               let modelInput = TapDetectorInput(input_1: modelInputArray)
               let prediction = try model.prediction(input: modelInput)
               let label = prediction.classLabel // L'etichetta predetta (es. "tap" o "sfondo").
               
               if let confidence = prediction.var_69[label] {
                   let confidencePercent = Int(confidence * 100)
                   let confidenceThreshold: Double = 1.9 // Soglia di confidenza per considerare valida la predizione.

                   // Se viene rilevato un "tap" con confidenza sufficiente e non c'è un cooldown attivo.
                   if label == "tap" && confidence >= confidenceThreshold && !isActionOnCooldown {
                       
                       updateStatus("TAP! (\(confidencePercent)%)")
                       UIImpactFeedbackGenerator(style: .heavy).impactOccurred() // Feedback aptico.
                       
                       // Attiva il cooldown per evitare rilevamenti duplicati.
                       isActionOnCooldown = true
                       DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) { self.isActionOnCooldown = false }
                   } else if !isActionOnCooldown {
                       updateStatus("In attesa del tap...")
                   }
               }
           } catch {
               print("❌ Errore di predizione: \(error)")
           }
       }
    
        // MARK: - Azioni e Aggiornamenti UI
    
        /// Chiamato quando l'utente cambia il segmento nel modeSelector.
        @objc private func modeChanged(_ sender: UISegmentedControl) {
            currentMode = sender.selectedSegmentIndex == 0 ? .recording : .inference
        }
        
        /// Aggiorna la visibilità degli elementi UI in base alla modalità corrente.
        private func updateUIVisibility() {
            let isRecordingMode = (currentMode == .recording)
            
            // Mostra/nasconde i pulsanti di registrazione.
            let recordingControls = view.subviews.filter { $0 is UIStackView }
            recordingControls.forEach { $0.isHidden = !isRecordingMode }
            
            // Mostra/nasconde la barra di avanzamento.
            progressView.isHidden = !isRecordingMode
            
            // Aggiorna il messaggio di stato.
            if isRecordingMode {
                updateStatus("Pronto a registrare.")
            } else {
                updateStatus("Inference Mode: Attivo.")
            }
        }
    
    /// Configura i pulsanti per avviare la registrazione di una finestra "tap" o "sfondo".
    private func setupRecordingControls() {
            let recordTapButton = UIButton.createStyledButton(title: "Registra Finestra TAP", color: .systemBlue, tag: 0)
            recordTapButton.addTarget(self, action: #selector(triggerWindowRecording(_:)), for: .touchUpInside)
            
            let recordBgButton = UIButton.createStyledButton(title: "Registra Finestra SFONDO", color: .systemGray, tag: 1)
            recordBgButton.addTarget(self, action: #selector(triggerWindowRecording(_:)), for: .touchUpInside)
            
            // Usa una StackView per organizzare i pulsanti.
            let stackView = UIStackView(arrangedSubviews: [recordTapButton, recordBgButton])
            stackView.translatesAutoresizingMaskIntoConstraints = false
            stackView.axis = .vertical
            stackView.spacing = 15
            stackView.distribution = .fillEqually
            view.addSubview(stackView)
            
            NSLayoutConstraint.activate([
                stackView.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 20),
                stackView.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -20),
                stackView.bottomAnchor.constraint(equalTo: statusLabel.topAnchor, constant: -15),
                stackView.heightAnchor.constraint(equalToConstant: 110)
            ])
        }
    
    // MARK: - Azioni di Registrazione
    
       /// Avvia il processo di registrazione di una singola finestra di dati.
       @objc private func triggerWindowRecording(_ sender: UIButton) {
           // Impedisce di avviare una nuova registrazione se una è già in corso.
           guard !isRecordingWindow else {
               showAlert(title: "Registrazione in corso", message: "Attendi il completamento della finestra corrente.")
               return
           }
           
           // Resetta lo stato per una registrazione pulita.
           self.lastProcessedLandmarks = nil
           self.lastRelativeYVelocity = nil
           
           isRecordingWindow = true
           windowBuffer = [] // Svuota il buffer.
           progressView.setProgress(0.0, animated: false)
           // Imposta l'etichetta in base al pulsante premuto (tag 0 per "tap", 1 per "sfondo").
           currentRecordingLabel = sender.tag == 0 ? "tap" : "sfondo"
           updateStatus("🔴 Registrazione finestra '\(currentRecordingLabel)'...")
       }
    
    /// Salva il buffer della finestra completata in un file JSON.
    private func saveWindowToFile() {
            // Crea un nome file univoco con timestamp.
            let formatter = DateFormatter()
            formatter.dateFormat = "yyyyMMdd_HHmmss_SSS"
            let fileName = "\(currentRecordingLabel)_window_\(formatter.string(from: Date())).json"
            
            let dataToSave = self.windowBuffer
            
            do {
                // Codifica i dati in formato JSON.
                let jsonData = try JSONEncoder().encode(dataToSave)
                if let documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
                    let fileURL = documentsDirectory.appendingPathComponent(fileName)
                    // Scrive i dati sul disco.
                    try jsonData.write(to: fileURL)
                    print("💾 Finestra salvata: \(fileName)")
                }
            } catch {
                print("❌ Errore durante il salvataggio: \(error)")
                // Feedback visivo di errore sulla progress bar.
                DispatchQueue.main.async {
                    self.progressView.progressTintColor = .systemRed
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
                        self.progressView.progressTintColor = .systemBlue
                    }
                }
                return
            }
            
            // Feedback visivo di successo.
            DispatchQueue.main.async {
                self.progressView.progressTintColor = .systemGreen
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
                    self.progressView.progressTintColor = .systemBlue
                    self.progressView.setProgress(0.0, animated: false)
                    self.updateStatus("✅ Finestra salvata. Pronto per la prossima.")
                }
            }
        }
    
    // MARK: - Metodi Helper
    
    /// Chiamato quando si preme il pulsante di chiusura. Esegue la callback onDismiss.
    @objc private func closeButtonTapped() {
        onDismiss?()
    }
    
    /// Gestisce l'accensione e lo spegnimento della torcia del dispositivo.
    @objc private func toggleTorch(forceOff: Bool = false) {
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back), device.hasTorch else {
            showAlert(title: "Torcia non disponibile", message: "Il tuo dispositivo non supporta la torcia.")
            return
        }

        do {
            try device.lockForConfiguration()
            
            if forceOff {
                device.torchMode = .off
                isTorchOn = false
            } else {
                device.torchMode = isTorchOn ? .off : .on
                isTorchOn.toggle()
            }
            
            device.unlockForConfiguration()
        } catch {
            print("❌ Errore durante la configurazione della torcia: \(error)")
        }
        
        // Aggiorna l'icona del pulsante.
        let iconName = isTorchOn ? "bolt.fill" : "bolt.slash.fill"
        torchButton.setImage(UIImage(systemName: iconName), for: .normal)
    }

    /// Aggiorna il testo dell'etichetta di stato sul thread principale.
    private func updateStatus(_ message: String) {
        DispatchQueue.main.async {
            self.statusLabel.text = message
        }
    }

    /// Mostra un semplice alert con un messaggio.
    private func showAlert(title: String, message: String) {
        DispatchQueue.main.async {
            let alert = UIAlertController(title: title, message: message, preferredStyle: .alert)
            alert.addAction(UIAlertAction(title: "OK", style: .default))
            self.present(alert, animated: true)
        }
    }
    
    /// Configura il pulsante della torcia.
    private func setupTorchButton() {
        torchButton = UIButton(type: .system)
        let iconName = isTorchOn ? "bolt.fill" : "bolt.slash.fill"
        torchButton.setImage(UIImage(systemName: iconName), for: .normal)
        torchButton.tintColor = .white
        torchButton.backgroundColor = UIColor.black.withAlphaComponent(0.7)
        torchButton.layer.cornerRadius = 20
        torchButton.translatesAutoresizingMaskIntoConstraints = false
        torchButton.addTarget(self, action: #selector(toggleTorch), for: .touchUpInside)
        view.addSubview(torchButton)
        
        NSLayoutConstraint.activate([
            torchButton.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 70),
            torchButton.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -70),
            torchButton.widthAnchor.constraint(equalToConstant: 40),
            torchButton.heightAnchor.constraint(equalToConstant: 40)
        ])
    }
    
    /// Configura il pulsante di chiusura.
    private func setupCloseButton() {
        let closeButton = UIButton(type: .system)
        closeButton.setTitle("✕", for: .normal)
        closeButton.titleLabel?.font = UIFont.systemFont(ofSize: 24, weight: .bold)
        closeButton.tintColor = .white
        closeButton.backgroundColor = UIColor.black.withAlphaComponent(0.7)
        closeButton.layer.cornerRadius = 20
        closeButton.translatesAutoresizingMaskIntoConstraints = false
        closeButton.addTarget(self, action: #selector(closeButtonTapped), for: .touchUpInside)
        view.addSubview(closeButton)
        
        NSLayoutConstraint.activate([
            closeButton.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 70),
            closeButton.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -20),
            closeButton.widthAnchor.constraint(equalToConstant: 40),
            closeButton.heightAnchor.constraint(equalToConstant: 40)
        ])
    }
    
    /// Gestisce la rotazione dell'interfaccia, aggiornando l'orientamento del video.
    override func viewWillTransition(to size: CGSize, with coordinator: UIViewControllerTransitionCoordinator) {
        super.viewWillTransition(to: size, with: coordinator)
        
        coordinator.animate(alongsideTransition: { [weak self] _ in
            guard let self = self, let connection = self.previewLayer?.connection else { return }
            
            let orientation = self.view.window?.windowScene?.interfaceOrientation ?? .portrait
            
            // Imposta l'angolo di rotazione corretto per il video.
            switch orientation {
            case .portrait:
                connection.videoRotationAngle = 90
            case .portraitUpsideDown:
                connection.videoRotationAngle = 270
            case .landscapeLeft:
                connection.videoRotationAngle = 180
            case .landscapeRight:
                connection.videoRotationAngle = 0
            default:
                connection.videoRotationAngle = 90
            }
        })
    }
}

// MARK: - Delegati

/// Estensione per conformarsi al delegato dell'output video.
extension HandDetectionViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    /// Questo metodo viene chiamato per ogni frame catturato dalla fotocamera.
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let handLandmarker = handLandmarker else { return }
        
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }
        
        // Ottiene il timestamp del frame in millisecondi.
        let timestamp = Int(CMSampleBufferGetPresentationTimeStamp(sampleBuffer).seconds * 1000)
        
        // Crea un oggetto MPImage, il formato richiesto da MediaPipe.
        guard let mpImage = try? MPImage(pixelBuffer: pixelBuffer, orientation: .up) else {
            return
        }
        
        // Esegue il rilevamento in modo asincrono.
        try? handLandmarker.detectAsync(image: mpImage, timestampInMilliseconds: timestamp)
    }
}

/// Estensione per conformarsi al delegato dei risultati di HandLandmarker.
extension HandDetectionViewController: HandLandmarkerLiveStreamDelegate {
    /// Questo metodo viene chiamato quando MediaPipe ha finito di processare un'immagine.
    func handLandmarker(_ handLandmarker: HandLandmarker, didFinishDetection result: HandLandmarkerResult?, timestampInMilliseconds: Int, error: Error?) {
        // Torna sul thread principale per aggiornare la UI e processare i risultati.
        DispatchQueue.main.async {
            if let error = error {
                print("HandLandmarker error: \(error.localizedDescription)")
                self.handSkeletonView.updatePoints([]) // Pulisce lo scheletro in caso di errore.
                return
            }
            guard let result = result, let previewLayer = self.previewLayer else {
                self.handSkeletonView.updatePoints([]) // Pulisce lo scheletro se non ci sono risultati.
                return
            }

            // Converte le coordinate dei landmark per la visualizzazione.
            let convertedPoints = result.landmarks.flatMap { handLandmarks in
                handLandmarks.map { landmark in
                    let captureDevicePoint = CGPoint(x: CGFloat(landmark.x), y: CGFloat(landmark.y))
                    return previewLayer.layerPointConverted(fromCaptureDevicePoint: captureDevicePoint)
                }
            }
            
            // Aggiorna la vista dello scheletro e processa i risultati.
            self.handSkeletonView.updatePoints(convertedPoints)
            self.processHandResults(result)
        }
    }
}

// MARK: - Estensione UIButton
/// Una piccola estensione per creare pulsanti con uno stile predefinito.
extension UIButton {
    static func createStyledButton(title: String, color: UIColor, tag: Int = 0) -> UIButton {
        let button = UIButton(type: .system)
        button.setTitle(title, for: .normal)
        button.tintColor = .white
        button.backgroundColor = color
        button.layer.cornerRadius = 8
        button.tag = tag
        return button
    }
}
