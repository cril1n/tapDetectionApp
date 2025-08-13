import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import coremltools as ct

# --- 1. CONFIGURAZIONE ---
# Definizione dei parametri fondamentali per l'addestramento e la struttura del modello
DATA_DIR = 'dataset/train'         
WINDOW_SIZE = 25                    # Numero di campioni per ogni sequenza temporale
NUM_FEATURES = 3                    # Numero di caratteristiche per ogni campione (velocità, accelerazione, ratio)
NUM_CLASSES = 2                     # Numero di classi da classificare (sfondo, tap)
BATCH_SIZE = 16                     # Dimensione del batch per l'addestramento
NUM_EPOCHS = 500                    # Numero di iterazioni complete sul dataset
LEARNING_RATE = 0.0001              # Tasso di apprendimento

# --- 2. CLASSE PER CARICARE IL DATASET ---
class HandGestureDataset(Dataset):
    """
    Classe personalizzata per caricare e preprocessare i dati dei gesti delle mani.
    Estende la classe Dataset di PyTorch per permettere l'uso del DataLoader.
    """
    def __init__(self, data_dir):
        # Inizializzazione delle liste per file di dati ed etichette
        self.data_files = []
        self.labels = []
        # Mappa che associa le classi agli indici numerici
        self.label_map = {'sfondo': 0, 'tap': 1}

        # Caricamento dei dati da ogni cartella di classe
        for label_name, label_idx in self.label_map.items():
            class_dir = os.path.join(data_dir, label_name)
            if not os.path.isdir(class_dir):
                print(f"Attenzione: La cartella {class_dir} non esiste.")
                continue
            # Raccolta di tutti i file JSON nella cartella della classe
            for filename in os.listdir(class_dir):
                if filename.endswith('.json'):
                    self.data_files.append(os.path.join(class_dir, filename))
                    self.labels.append(label_idx)

    def __len__(self):
        # Restituisce il numero totale di campioni nel dataset
        return len(self.data_files)

    def __getitem__(self, idx):
        # Carica un singolo campione in base all'indice
        file_path = self.data_files[idx]
        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                # Gestione degli errori di decodifica JSON
                print(f"Errore nel leggere il file JSON: {file_path}")
                return torch.zeros(NUM_FEATURES, WINDOW_SIZE), torch.tensor(0, dtype=torch.long)

        # Estrazione delle caratteristiche dai dati JSON
        velocity = [item.get('relativeYVelocity', 0.0) for item in data]          # Velocità relativa verticale
        acceleration = [item.get('relativeYAcceleration', 0.0) for item in data]  # Accelerazione relativa verticale
        velocity_ratio = [item.get('stabilityRatio', 0.0) for item in data]       # Rapporto di stabilità
        
        # Organizzazione delle caratteristiche in un array 
        features = np.array([velocity, acceleration, velocity_ratio], dtype=np.float32)
        
        # Verifica della forma corretta dei dati
        if features.shape != (NUM_FEATURES, WINDOW_SIZE):
             print(f"Attenzione: Il file {file_path} ha una forma errata {features.shape}. Verrà saltato.")
             # Passa al campione successivo in caso di forma errata
             return self.__getitem__((idx + 1) % len(self))

        # Conversione in tensori PyTorch
        features_tensor = torch.from_numpy(features)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return features_tensor, label_tensor

# --- 3. DEFINIZIONE DEL MODELLO CNN-LSTM ---
class CNNLSTM(nn.Module):
    """
    Modello ibrido che combina CNN (per l'estrazione di feature spaziali) 
    e LSTM (per catturare dipendenze temporali nei dati sequenziali).
    """
    def __init__(self, num_features, num_classes, lstm_hidden_size=64):
        super(CNNLSTM, self).__init__()
        
        # Layer CNN per l'estrazione delle caratteristiche
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()  # Funzione di attivazione non lineare
        self.pool = nn.MaxPool1d(kernel_size=2)  # Riduce la dimensionalità e aiuta a evitare l'overfitting
        
        lstm_input_size = 32  # Dimensione di input per LSTM (corrisponde a out_channels della CNN)
        # Layer LSTM per catturare le dipendenze temporali
        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=lstm_hidden_size, batch_first=True)
        # Fully connected layer per la classificazione finale
        self.fc = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x):
        # Passaggio attraverso la CNN
        x = self.conv1(x)  # Applica convoluzione 1D
        x = self.relu(x)   # Applica funzione di attivazione
        x = self.pool(x)   # Applica max pooling
        
        # Riorganizzazione del tensore per l'input LSTM (batch, seq_len, features)
        x = x.permute(0, 2, 1) 
        
        # Passaggio attraverso LSTM
        lstm_out, _ = self.lstm(x)
        # Prendiamo solo l'ultimo stato nascosto per la classificazione
        last_hidden_state = lstm_out[:, -1, :]
        
        # Classificazione finale
        out = self.fc(last_hidden_state)
        return out

# --- 4. TRAINING LOOP ---
print("Caricamento dati...")
# Inizializzazione del dataset di addestramento
train_dataset = HandGestureDataset(data_dir=DATA_DIR)
if len(train_dataset) == 0:
    print("Errore: Nessun dato trovato nella cartella di training. Controlla il percorso e la struttura delle cartelle.")
    exit()

# Creazione del DataLoader per l'addestramento batch per batch
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

print("Inizializzazione modello...")
# Creazione del modello, funzione di perdita e ottimizzatore
model = CNNLSTM(num_features=NUM_FEATURES, num_classes=NUM_CLASSES)
criterion = nn.CrossEntropyLoss()  # Funzione di perdita standard per classificazione
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  # Ottimizzatore Adam

print("Inizio training...")
# Loop principale di addestramento
for epoch in range(NUM_EPOCHS):
    for i, (sequences, labels) in enumerate(train_loader):
        # Forward pass: calcolo delle previsioni
        outputs = model(sequences)
        # Calcolo della perdita
        loss = criterion(outputs, labels)
        
        # Backward pass e ottimizzazione
        optimizer.zero_grad()  # Azzera i gradienti per la nuova iterazione
        loss.backward()        # Calcola i gradienti
        optimizer.step()       # Aggiorna i pesi del modello
        
    # Stampa dei progressi dopo ogni epoca
    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}')

print("Training completato!")

# Salvataggio dei pesi del modello in formato PyTorch
torch.save(model.state_dict(), "tap_model_weights.pth")
print("Pesi del modello PyTorch salvati in tap_model_weights.pth")

# --- 5. CONVERSIONE IN CORE ML (METODO CORRETTO E SEMPLIFICATO) ---
print("Conversione del modello in Core ML...")
# Imposta il modello in modalità valutazione (disattiva dropout, etc.)
model.eval() 

# Crea un input fittizio per tracciare il modello
dummy_input = torch.rand(1, NUM_FEATURES, WINDOW_SIZE) 
# Crea una versione tracciata del modello (più ottimizzata)
traced_model = torch.jit.trace(model, dummy_input)

# 1. Definisci le etichette delle classi per il modello Core ML
class_labels = ['sfondo', 'tap']

# 2. Crea la configurazione del classificatore
classifier_config = ct.ClassifierConfig(class_labels)

# 3. Conversione in formato Core ML
# CoreMLTools aggiungerà automaticamente un layer Softmax e configurerà gli output
mlmodel = ct.convert(
    traced_model,
    inputs=[ct.TensorType(name="input_1", shape=dummy_input.shape)],
    classifier_config=classifier_config,
    convert_to="neuralnetwork"  # Usa il formato neural network invece di ML Program
)

# Salvataggio del modello Core ML
mlmodel.save("TapDetector.mlmodel")

# Stampa informazioni sul modello convertito
print("\nModello salvato come TapDetector.mlmodel!")
print("Input atteso:", mlmodel.get_spec().description.input[0].name)
print("Output di probabilità:", mlmodel.get_spec().description.predictedProbabilitiesName)
print("Output etichetta:", mlmodel.get_spec().description.predictedFeatureName)