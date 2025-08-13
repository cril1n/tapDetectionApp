import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Importa le classi del modello e del dataset dal tuo script di training
from train_model import CNNLSTM, HandGestureDataset 

# --- 1. CONFIGURAZIONE ---
TEST_DATA_DIR = 'dataset/test' # Usa la cartella di test!
MODEL_WEIGHTS_PATH = 'tap_model_weights.pth' # File dei pesi salvato
WINDOW_SIZE = 25
NUM_FEATURES = 3
NUM_CLASSES = 2
BATCH_SIZE = 8

# --- 2. CARICAMENTO DATI E MODELLO ---
print("Caricamento dati di test...")
test_dataset = HandGestureDataset(data_dir=TEST_DATA_DIR)
if len(test_dataset) == 0:
    print(f"Errore: Nessun dato trovato nella cartella di test: {TEST_DATA_DIR}")
    exit()
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Caricamento modello addestrato...")
model = CNNLSTM(num_features=NUM_FEATURES, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH))
model.eval() # Imposta il modello in modalità valutazione (molto importante!)

# --- 3. ESECUZIONE DELLA VALUTAZIONE ---
all_labels = []
all_predictions = []

print("Esecuzione delle predizioni sul test set...")
with torch.no_grad(): # Disabilita il calcolo dei gradienti per velocizzare
    for sequences, labels in test_loader:
        outputs = model(sequences)
        # Ottieni le predizioni prendendo la classe con la probabilità più alta
        _, predicted = torch.max(outputs.data, 1)
        
        all_labels.extend(labels.numpy())
        all_predictions.extend(predicted.numpy())

# --- 4. CALCOLO E STAMPA DELLE METRICHE ---
# Converte le etichette numeriche in nomi per una migliore leggibilità
class_names = ['sfondo', 'tap']
all_labels_named = [class_names[i] for i in all_labels]
all_predictions_named = [class_names[i] for i in all_predictions]

# Accuratezza
accuracy = accuracy_score(all_labels, all_predictions)
print("\n--- Risultati della Valutazione ---")
print(f"✅ Accuratezza Totale: {accuracy * 100:.2f}%")

# Matrice di Confusione
print("\n🌀 Matrice di Confusione:")
print("Indica quanti campioni di una classe sono stati classificati come un'altra.")
print("Righe = Verità | Colonne = Predizione")
cm = confusion_matrix(all_labels_named, all_predictions_named, labels=class_names)
print(cm)

# Report di Classificazione
print("\n📊 Report di Classificazione Dettagliato:")
report = classification_report(all_labels_named, all_predictions_named, target_names=class_names)
print(report)