import json
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

def plot_motion_data(file_path, save_dir=None):
    """
    Legge un file JSON di dati di movimento e ne disegna il grafico.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Errore: File non trovato a '{file_path}'")
        return
    except json.JSONDecodeError:
        print(f"Errore: Il file '{file_path}' non è un JSON valido.")
        return

    # Converte i dati in un DataFrame di pandas per una facile manipolazione
    df = pd.DataFrame(data)

    if 'relativeZVelocity' not in df.columns or 'relativeZAcceleration' not in df.columns:
        print("Errore: Il JSON non contiene le chiavi 'relativeZVelocity' o 'relativeZAcceleration'.")
        return

    # Crea il grafico
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['relativeZVelocity'], label='Velocità Z Relativa', color='blue', marker='o')
    plt.plot(df.index, df['relativeZAcceleration'], label='Accelerazione Z Relativa', color='red', marker='x', linestyle='--')

    # Aggiunge dettagli al grafico
    filename = Path(file_path).stem
    plt.title(f'Profilo del Gesto - {filename}')
    plt.xlabel('Frame (Tempo)')
    plt.ylabel('Valore Normalizzato')
    plt.legend()
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.5) # Linea dello zero per riferimento
    
    # Salva il grafico
    if save_dir:
        output_path = os.path.join(save_dir, f"{filename}_plot.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Grafico salvato: {output_path}")
    
    plt.close()  # Chiude la figura per liberare memoria

def process_all_json_files():
    """
    Processa tutti i file JSON nella cartella dello script e salva i grafici.
    """
    # Ottiene la cartella dello script
    script_dir = Path(__file__).parent
    
    # Trova tutti i file JSON nella cartella
    json_files = list(script_dir.glob("*.json"))
    
    if not json_files:
        print("Nessun file JSON trovato nella cartella dello script.")
        return
    
    # Crea una cartella per i grafici se non esiste
    output_dir = script_dir / "grafici"
    output_dir.mkdir(exist_ok=True)
    
    print(f"Trovati {len(json_files)} file JSON. Elaborazione in corso...")
    
    # Processa ogni file JSON
    for json_file in json_files:
        print(f"Elaborando: {json_file.name}")
        plot_motion_data(str(json_file), str(output_dir))
    
    print(f"\nElaborazione completata! I grafici sono stati salvati in: {output_dir}")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Se non ci sono argomenti, processa tutti i file JSON nella cartella
        process_all_json_files()
    elif len(sys.argv) == 2:
        # Se c'è un argomento, processa solo quel file (comportamento originale)
        plot_motion_data(sys.argv[1])
        plt.show()
    else:
        print("Utilizzo:")
        print("  python visualize.py                    # Processa tutti i JSON nella cartella")
        print("  python visualize.py <file.json>        # Processa un singolo file")