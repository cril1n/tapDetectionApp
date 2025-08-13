import json
import pandas as pd
import matplotlib.pyplot as plt
import sys

def plot_motion_data(file_path):
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
    plt.title(f'Profilo del Gesto - {file_path.split("/")[-1]}')
    plt.xlabel('Frame (Tempo)')
    plt.ylabel('Valore Normalizzato')
    plt.legend()
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.5) # Linea dello zero per riferimento
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Utilizzo: python visualize.py <percorso_del_tuo_file.json>")
    else:
        plot_motion_data(sys.argv[1])