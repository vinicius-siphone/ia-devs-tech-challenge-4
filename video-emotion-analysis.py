# Importação das bibliotecas
import cv2
from deepface import DeepFace
import os
import csv
from tqdm import tqdm
import numpy as np

# Emoções associadas a risco psicológico
NEGATIVE_EMOTIONS = ["fear", "sad", "angry", "disgust"]

# Peso das emoções
EMOTION_WEIGHTS = {
    "fear": 3,
    "sad": 2,
    "angry": 2,
    "disgust": 2,
    "neutral": 0,
    "happy": -1,
    "surprise": 1
}

# Cálculo do score de risco psicológico com base no histórico de emoções
def calculate_risk_score(emotions_history):
    score = 0

    for emo in emotions_history:
        score += EMOTION_WEIGHTS.get(emo, 0)

    avg_score = score / len(emotions_history)

    return round(avg_score, 2)


# Função principal para detectar emoções no vídeo e gerar relatório
def detect_emotions(video_path, output_path, report_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    emotions_history = []

    # Arquivo CSV de relatório
    with open(report_path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Frame",
                         "Dominant Emotion",
                         "Fear",
                         "Sad",
                         "Angry",
                         "Happy",
                         "Neutral",
                         "Risk Score"
                         ])

        for frame_id in tqdm(range(total_frames), desc="Processando vídeo"):
            ret, frame = cap.read()

            if not ret:
                break

            try:
                result = DeepFace.analyze(
                    frame,
                    actions=["emotion"],
                    enforce_detection=False
                )
            except:
                continue

            for face in result:
                x, y, w, h = face["region"]["x"], face["region"]["y"], face["region"]["w"], face["region"]["h"]

                emotions = face["emotion"]
                dominant = face["dominant_emotion"]

                emotions_history.append(dominant)

                # Mantém o histórico recente (últimos 5 segundos)
                max_history = fps * 5
                if len(emotions_history) > max_history:
                    emotions_history.pop(0)

                risk = calculate_risk_score(emotions_history)

                # Define alerta
                alert = ""
                color = (0, 255, 0)

                if risk >= 1.5:
                    alert = "ALERTA: Estresse"
                    color = (0, 165, 255)

                if risk >= 2.5:
                    alert = "ALERTA: Possível Trauma"
                    color = (0, 0, 255)

                # Desenha o rosto no vídeo
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

                # Emoção
                cv2.putText(frame, f"{dominant}", (x, y-40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Score
                cv2.putText(
                    frame, f"Risk: {risk}", (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Alerta
                if alert:
                    cv2.putText(frame, alert, (x, y+h+25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                writer.writerow([
                    frame_id,
                    dominant,
                    emotions.get("fear", 0),
                    emotions.get("sad", 0),
                    emotions.get("angry", 0),
                    emotions.get("happy", 0),
                    emotions.get("neutral", 0),
                ])
            out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("Processamento concluído.")
    print("Relatório salvo em:", report_path)


# Paths
base_dir = os.path.dirname(os.path.abspath(__file__))

input_video = os.path.join(base_dir, "video.mp4")
output_video = os.path.join(base_dir, "output_video.mp4")
report_file = os.path.join(base_dir, "emotion_report.csv")

detect_emotions(input_video, output_video, report_file)