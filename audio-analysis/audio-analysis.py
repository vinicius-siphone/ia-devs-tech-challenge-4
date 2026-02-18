# Importações de bibliotecas necessarias
import json
import os
import re
import unicodedata
from typing import Any, Dict, List, Tuple

# Bibliotecas adicionais para processamento de linguagem natural, análise de tópicos e manipulação de áudio
import gensim
import nltk
import numpy as np
import speech_recognition as sr
from gensim import corpora
from nltk.corpus import stopwords
from pydub import AudioSegment

# Configurações e constantes para o processamento do áudio e análise de risco
AUDIO_FILE = "audio-analysis.wav"
TRANSCRIPT_FILE = "transcricao-audio.txt"
REPORT_FILE = "relatorio_audio.json"

LANGUAGE = "pt-BR"
CHUNK_SECONDS = 30
LDA_TOPICS = 4
LDA_PASSES = 12

CHUNK_RISK_MEDIUM_THRESHOLD = 4
CHUNK_RISK_HIGH_THRESHOLD = 8
OVERALL_RISK_MEDIUM_THRESHOLD = 1.0
OVERALL_RISK_HIGH_THRESHOLD = 2.0
STRONG_SIGNAL_COVERAGE_THRESHOLD = 0.25

SPEAKER_MAP = {
    "SPEAKER_00": "Paciente",
    "SPEAKER_01": "Profissional",
}
ANALYSIS_ROLE = "Paciente"

# Configurações de stopwords e regex para tokenização
nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("portuguese"))
TOKEN_RE = re.compile(r"[a-zà-ú]+", re.IGNORECASE)

#  Lexico de risco com categorias, pesos e termos associados
RISK_LEXICON = {
    "violence_abuse": {
        "peso": 3,
        "termos": {
            "agressao", "agressão", "violencia", "violência", "abuso", "ameaça", "ameaca",
            "coacao", "coação", "medo", "machucou", "machucar", "controle", "controlar",
            "humilhação", "humilhacao", "chantagem", "assédio", "assedio", "agride", "bate"
        },
    },
    "trauma_stress": {
        "peso": 2,
        "termos": {
            "trauma", "traumático", "traumatico", "pânico", "panico", "crise", "ansiedade",
            "pesadelo", "gatilho", "flashback", "tremor", "hipervigilancia", "hipervigilância", "vitima"
        },
    },
    "depression_low_mood": {
        "peso": 1,
        "termos": {
            "depressao", "depressão", "triste", "desanimado", "culpa", "vazio", "apatia", "choro", "chorar"
        },
    },
}


# Classificar o nível de risco com base no score e nos limiares definidos
def classify_risk_level(score: float, medium_threshold: float, high_threshold: float) -> str:
    if score >= high_threshold:
        return "alto"
    if score >= medium_threshold:
        return "medio"
    return "baixo"


# Normalizar um termo para sua forma canônica, removendo acentos e convertendo para minúsculas
def canonicalize_term(term: str) -> str:
    normalized = unicodedata.normalize("NFD", (term or "").lower())
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


# Converter segundos para formato mm:ss
def seconds_to_mmss(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


# Normalizar o texto para tokens, removendo stopwords e palavras muito curtas
def normalize_tokens(text: str) -> List[str]:
    tokens = TOKEN_RE.findall((text or "").lower())
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 2]


# Heuristica simples para dividir o texto em falas/utterances, usando pontuação como delimitador
def split_utterances(text: str) -> List[str]:
    parts = re.split(r"(?<=[\.\!\?\;\:])\s+", (text or "").strip())
    return [p.strip() for p in parts if p and p.strip()]


# Dividir o áudio em chunks de N segundos, retornando uma lista de tuplas com o áudio do chunk e seus timestamps
def split_audio(audio_path: str, chunk_seconds: int = CHUNK_SECONDS) -> List[Tuple[AudioSegment, float, float]]:
    audio = AudioSegment.from_file(audio_path)
    total_ms = len(audio)
    chunk_ms = chunk_seconds * 1000

    chunks = []
    for i in range(0, total_ms, chunk_ms):
        start_ms = i
        end_ms = min(i + chunk_ms, total_ms)
        chunks.append(
            (audio[start_ms:end_ms], start_ms / 1000.0, end_ms / 1000.0))
    return chunks


# Transcrever o áudio em chunks, retornando uma lista de dicionários com texto e timestamps
def transcribe_in_chunks(
    audio_path: str,
    language: str = LANGUAGE,
    chunk_seconds: int = CHUNK_SECONDS,
) -> List[Dict[str, Any]]:
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True

    chunk_data = split_audio(audio_path, chunk_seconds=chunk_seconds)
    total_chunks = len(chunk_data)
    results: List[Dict[str, Any]] = []

    print(f"[Áudio] Iniciando transcrição em chunks de {chunk_seconds}s...")

    for idx, (chunk_audio, start_s, end_s) in enumerate(chunk_data, start=1):
        print(
            f"[Áudio] Chunk {idx}/{total_chunks} carregado ({seconds_to_mmss(start_s)} - {seconds_to_mmss(end_s)})")

        tmp_path = f"__tmp_chunk_{idx}.wav"
        chunk_audio.export(tmp_path, format="wav")

        with sr.AudioFile(tmp_path) as source:
            audio_data = recognizer.record(source)

        try:
            text = recognizer.recognize_google(audio_data, language=language)
        except sr.UnknownValueError:
            text = ""
        except sr.RequestError as exc:
            results.append({
                "chunk_index": idx,
                "start_sec": start_s,
                "end_sec": end_s,
                "text": "",
                "error": f"RequestError: {str(exc)}",
            })
            os.remove(tmp_path)
            print(
                "[Áudio] Erro ao acessar o servico de transcrição. Vou salvar o que ja foi processado.")
            break

        results.append({
            "chunk_index": idx,
            "start_sec": start_s,
            "end_sec": end_s,
            "text": text,
        })

        print(
            f"[Áudio] Chunk {idx}/{total_chunks} processado: {'ok' if text else 'sem fala reconhecida'}")
        os.remove(tmp_path)

    print(
        f"[Áudio] Transcrição em chunks finalizada. Total de chunks: {len(results)}")
    return results


# Heuristica simples para atribuir falas alternando entre os speakers definidos
def assign_speakers_heuristic(
    chunks: List[Dict[str, Any]],
    speaker_map: Dict[str, str] = None,
) -> List[Dict[str, Any]]:
    if speaker_map is None:
        speaker_map = SPEAKER_MAP

    speaker_ids = list(speaker_map.keys())
    turn_idx = 0
    diarized_turns: List[Dict[str, Any]] = []

    for chunk in chunks:
        text = (chunk.get("text", "") or "").strip()
        if not text:
            chunk["speaker_turns"] = []
            continue

        utterances = split_utterances(text)
        if not utterances:
            chunk["speaker_turns"] = []
            continue

        turns_for_chunk = []
        for utterance in utterances:
            speaker_id = speaker_ids[turn_idx % len(speaker_ids)]
            turn_idx += 1

            turn = {
                "chunk_index": chunk.get("chunk_index"),
                "start_sec": chunk.get("start_sec", 0),
                "end_sec": chunk.get("end_sec", 0),
                "speaker_id": speaker_id,
                "speaker_role": speaker_map.get(speaker_id, speaker_id),
                "text": utterance,
            }
            turns_for_chunk.append(turn)
            diarized_turns.append(turn)

        chunk["speaker_turns"] = turns_for_chunk

    return diarized_turns


# Gerar chunks filtrados por papel, mantendo somente as falas do paciente
def build_role_only_chunks(chunks: List[Dict[str, Any]], role: str = ANALYSIS_ROLE) -> List[Dict[str, Any]]:
    filtered = []
    for chunk in chunks:
        role_texts = [
            (turn.get("text", "") or "").strip()
            for turn in chunk.get("speaker_turns", [])
            if turn.get("speaker_role") == role and (turn.get("text", "") or "").strip()
        ]
        merged_text = " ".join(role_texts).strip()
        if not merged_text:
            continue

        filtered.append({
            "chunk_index": chunk.get("chunk_index"),
            "start_sec": chunk.get("start_sec", 0),
            "end_sec": chunk.get("end_sec", 0),
            "text": merged_text,
        })
    return filtered


# Gerar um resumo por papel de falante, incluindo contagem de turns, chars e score de risco agregado
def summarize_speakers(diarized_turns: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_speaker: Dict[str, Dict[str, Any]] = {}

    for turn in diarized_turns:
        role = turn.get("speaker_role", "Desconhecido")
        text = (turn.get("text", "") or "").strip()
        if role not in by_speaker:
            by_speaker[role] = {
                "speaker_role": role,
                "turns": 0,
                "chars": 0,
                "risk": {"score": 0, "level": "baixo", "hits": []},
            }
        by_speaker[role]["turns"] += 1
        by_speaker[role]["chars"] += len(text)

    for role in by_speaker.keys():
        merged_text = " ".join(
            t["text"] for t in diarized_turns if t.get("speaker_role") == role)
        by_speaker[role]["risk"] = risk_score_for_text(merged_text)

    return {
        "diarization_mode": "heuristic_speech",
        "speakers": sorted(by_speaker.values(), key=lambda x: x["speaker_role"]),
    }


# Gerar um txt com a transcrição completa e timestamps, incluindo os papéis dos falantes
def write_transcript_txt(chunks: List[Dict[str, Any]], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as handle:
        for chunk in chunks:
            start = seconds_to_mmss(chunk.get("start_sec", 0))
            end = seconds_to_mmss(chunk.get("end_sec", 0))
            speaker_turns = chunk.get("speaker_turns", [])

            if speaker_turns:
                for turn in speaker_turns:
                    role = turn.get("speaker_role", "Desconhecido")
                    text = (turn.get("text", "") or "").strip()
                    if text:
                        handle.write(f"[{start} - {end}] {role}: {text}\n")
                handle.write("\n")
            else:
                text = (chunk.get("text", "") or "").strip()
                if text:
                    handle.write(f"[{start} - {end}] {text}\n\n")


# Gerar tópicos LDA a partir dos chunks filtrados por papel (somente falas do paciente)
def lda_topics_from_chunks(chunks: List[Dict[str, Any]], num_topics: int = LDA_TOPICS, passes: int = LDA_PASSES) -> Dict[str, Any]:
    tokenized_docs = []
    for chunk in chunks:
        text = (chunk.get("text", "") or "").strip()
        if not text:
            continue
        tokens = normalize_tokens(text)
        if tokens:
            tokenized_docs.append(tokens)

    if len(tokenized_docs) < 3:
        return {
            "ok": False,
            "reason": "Texto insuficiente apos transcrição para treinar LDA",
            "topics": [],
        }

    dictionary = corpora.Dictionary(tokenized_docs)
    corpus = [dictionary.doc2bow(tokens) for tokens in tokenized_docs]

    lda = gensim.models.ldamodel.LdaModel(
        corpus=corpus,
        num_topics=num_topics,
        id2word=dictionary,
        passes=passes,
        random_state=42,
    )

    topics = []
    for topic_id in range(num_topics):
        top_words = lda.show_topic(topic_id, topn=10)
        topics.append({
            "topic_id": topic_id,
            "top_words": [{"word": w, "weight": float(p)} for w, p in top_words],
        })

    doc_topics = []
    for idx, bow in enumerate(corpus):
        dist = lda.get_document_topics(bow, minimum_probability=0.0)
        dominant_topic, dominant_prob = sorted(
            dist, key=lambda x: x[1], reverse=True)[0]
        doc_topics.append({
            "doc_index": idx,
            "dominant_topic": int(dominant_topic),
            "dominant_prob": float(dominant_prob),
            "topic_distribution": [{"topic": int(t), "prob": float(p)} for t, p in dist],
        })

    avg_probs = np.zeros(num_topics, dtype=float)
    for doc in doc_topics:
        for item in doc["topic_distribution"]:
            avg_probs[item["topic"]] += item["prob"]
    avg_probs /= max(len(doc_topics), 1)

    return {
        "ok": True,
        "topics": topics,
        "doc_topics": doc_topics,
        "overall_dominant_topic": int(np.argmax(avg_probs)),
        "overall_topic_probs": [{"topic": i, "prob": float(avg_probs[i])} for i in range(num_topics)],
    }

# Gerar um resumo dos assuntos discutidos com base nos tópicos LDA e suas presenças no áudio


def summarize_discussed_subjects(lda_report: Dict[str, Any], max_topics: int = 4) -> List[Dict[str, Any]]:
    if not lda_report.get("ok"):
        return []

    probs = sorted(lda_report.get("overall_topic_probs", []),
                   key=lambda x: x.get("prob", 0.0), reverse=True)
    top_probs = probs[:max_topics]

    subjects = []
    for item in top_probs:
        topic_id = item["topic"]
        topic_def = next((t for t in lda_report.get("topics", [])
                         if t.get("topic_id") == topic_id), None)
        if not topic_def:
            continue

        subjects.append({
            "topic_id": topic_id,
            "presence": round(float(item.get("prob", 0.0)), 3),
            "keywords": [w["word"] for w in topic_def.get("top_words", [])[:8]],
        })
    return subjects


# Gerar um score de risco para um texto baseado no lexico definido
def risk_score_for_text(text: str) -> Dict[str, Any]:
    lower_text = (text or "").lower()
    score = 0
    hits = []

    for category, cfg in RISK_LEXICON.items():
        weight = cfg["peso"]
        found_terms = [term for term in cfg["termos"] if term in lower_text]

        if found_terms:
            unique_terms = sorted(set(found_terms))
            category_score = weight * len(unique_terms)
            score += category_score
            hits.append({
                "category": category,
                "category_score": int(category_score),
                "terms_found": unique_terms,
            })

    level = classify_risk_level(
        score,
        medium_threshold=CHUNK_RISK_MEDIUM_THRESHOLD,
        high_threshold=CHUNK_RISK_HIGH_THRESHOLD,
    )
    return {"score": int(score), "level": level, "hits": hits}


# Agregar risco por categoria e termos, e calcular score medio e nivel geral
def risk_over_chunks(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_chunk = []
    total_score = 0
    category_stats: Dict[str, Dict[str, Any]] = {}
    term_stats: Dict[str, Dict[str, Any]] = {}

    for chunk in chunks:
        risk = risk_score_for_text(chunk.get("text", ""))
        chunk_index = chunk.get("chunk_index")
        total_score += risk["score"]

        by_chunk.append({
            "chunk_index": chunk_index,
            "start": seconds_to_mmss(chunk.get("start_sec", 0)),
            "end": seconds_to_mmss(chunk.get("end_sec", 0)),
            "risk": risk,
        })

        for hit in risk["hits"]:
            category = hit["category"]
            category_entry = category_stats.setdefault(
                category,
                {"category": category, "score_sum": 0,
                    "term_mentions": 0, "chunk_indices": set()},
            )
            category_entry["score_sum"] += hit["category_score"]
            category_entry["term_mentions"] += len(hit["terms_found"])
            if chunk_index is not None:
                category_entry["chunk_indices"].add(chunk_index)

            for term in hit["terms_found"]:
                canonical = canonicalize_term(term)
                term_entry = term_stats.setdefault(
                    canonical,
                    {
                        "canonical_term": canonical,
                        "display_term": term,
                        "category": category,
                        "mentions": 0,
                        "chunk_indices": set(),
                    },
                )
                term_entry["mentions"] += 1
                if chunk_index is not None:
                    term_entry["chunk_indices"].add(chunk_index)

    avg_score = total_score / max(len(chunks), 1)
    overall_level = classify_risk_level(
        avg_score,
        medium_threshold=OVERALL_RISK_MEDIUM_THRESHOLD,
        high_threshold=OVERALL_RISK_HIGH_THRESHOLD,
    )

    categories_summary = sorted(
        [
            {
                "category": entry["category"],
                "score_sum": int(entry["score_sum"]),
                "term_mentions": int(entry["term_mentions"]),
                "chunk_count": len(entry["chunk_indices"]),
                "chunks": sorted(entry["chunk_indices"]),
            }
            for entry in category_stats.values()
        ],
        key=lambda x: (-x["score_sum"], -x["chunk_count"], x["category"]),
    )

    terms_summary = sorted(
        [
            {
                "term": entry["display_term"],
                "canonical_term": entry["canonical_term"],
                "category": entry["category"],
                "mentions": int(entry["mentions"]),
                "chunk_count": len(entry["chunk_indices"]),
                "chunks": sorted(entry["chunk_indices"]),
            }
            for entry in term_stats.values()
        ],
        key=lambda x: (-x["chunk_count"], -x["mentions"], x["canonical_term"]),
    )

    return {
        "overall_level": overall_level,
        "avg_score": round(avg_score, 2),
        "by_chunk": by_chunk,
        "categories_summary": categories_summary,
        "terms_summary": terms_summary,
    }


# Gerar uma análise geral a partir dos dados de risco encontrados
def build_analysis(chunks: List[Dict[str, Any]], risk_report: Dict[str, Any]) -> Dict[str, Any]:
    categories = {c["category"]: c for c in risk_report.get(
        "categories_summary", [])}
    terms = {t["canonical_term"]             : t for t in risk_report.get("terms_summary", [])}

    has_violence_abuse = "violence_abuse" in categories
    has_trauma = "trauma_stress" in categories
    has_depression = "depression_low_mood" in categories

    violence_mentions = terms.get("violencia", {}).get("chunk_count", 0)
    total_chunks = max(len(chunks), 1)
    sensitive_chunks = len([c for c in risk_report.get(
        "by_chunk", []) if c.get("risk", {}).get("score", 0) > 0])
    sensitive_coverage = round(sensitive_chunks / total_chunks, 3)

    overall_level = risk_report.get("overall_level")
    avg_score = risk_report.get("avg_score")

    dominant_category = None
    if categories:
        dominant_category = sorted(
            categories.values(),
            key=lambda x: (x.get("score_sum", 0), x.get("chunk_count", 0)),
            reverse=True,
        )[0]["category"]

    repeated_violence_signal = has_violence_abuse and (
        violence_mentions >= 2 or categories["violence_abuse"]["chunk_count"] >= 2
    )
    strong_sensitive_signal = overall_level in {
        "medio", "alto"} or sensitive_coverage >= STRONG_SIGNAL_COVERAGE_THRESHOLD

    if has_violence_abuse and (repeated_violence_signal or strong_sensitive_signal):
        label = "possivel_relato_violencia_ou_abuso"
        message = "Há indícios lexicais consistentes de violencia/abuso na transcrição."
    elif has_violence_abuse:
        label = "possivel_relato_violencia_ou_abuso"
        message = "Há indícios lexicais de violencia/abuso na transcrição."
    elif has_trauma and strong_sensitive_signal:
        label = "possiveis_sinais_trauma_estresse_relevantes"
        message = "Há indícios lexicais relevantes de trauma/estresse na transcrição."
    elif has_trauma:
        label = "possiveis_sinais_trauma_estresse"
        message = "Há indícios lexicais de trauma/estresse na transcrição."
    elif has_depression:
        label = "possiveis_sinais_humor_depressivo"
        message = "Há indícios lexicais de humor depressivo na transcrição."
    else:
        label = "sem_evidencia_lexical_relevante"
        message = "Nao foram encontrados indicios lexicais relevantes de violencia, abuso ou trauma."

    return {
        "classification": label,
        "message": message,
        "non_clinical_notice": "Resultado baseado em regras lexicais; nao representa diagnostico clínico.",
        "evidence": {
            "overall_level": overall_level,
            "avg_score": avg_score,
            "dominant_sensitive_category": dominant_category,
            "sensitive_chunks": sensitive_chunks,
            "total_chunks": total_chunks,
            "sensitive_chunk_coverage": sensitive_coverage,
            "violence_mentions_in_chunks": violence_mentions,
            "categories_detected": list(categories.keys()),
            "top_sensitive_terms": [t["term"] for t in risk_report.get("terms_summary", [])[:10]],
        },
    }


# Gerar um relatório a partir do report gerado
def get_audio_report(report: Dict[str, Any]) -> None:
    print("\nANÁLISE DO ÁUDIO (Falas do(a) Paciente)")
    print("_______________________________")

    lda = report.get("lda_topics", {})
    risk = report.get("risk_detection", {})
    final_summary = report.get("final_summary", {})

    if lda.get("ok"):
        print(
            f"\n[Topicos] Topico dominante geral: {lda.get('overall_dominant_topic')}")
        for item in sorted(lda.get("overall_topic_probs", []), key=lambda x: x["topic"]):
            topic_def = next((t for t in lda.get("topics", [])
                             if t.get("topic_id") == item["topic"]), None)
            if topic_def:
                words = ", ".join([w["word"]
                                  for w in topic_def["top_words"][:8]])
                print(
                    f" - Topico {item['topic']} | presença ~ {item['prob']:.2f} | palavras: {words}")
    else:
        print("\n[Topicos] Nao foi possivel gerar topicos (texto insuficiente).")

    print("\n[Sinais] Indicador (nao clínico) de temas sensiveis:")
    print(f" - Nivel geral: {risk.get('overall_level')}")
    print(f" - Score medio: {risk.get('avg_score')}")

    top_chunks = risk.get("top_chunks", [])
    if top_chunks:
        print("\n[Sinais] Trechos com maior evidencia (top):")
        for chunk in top_chunks:
            print(
                f" - [{chunk['start']} - {chunk['end']}] score={chunk['risk']['score']} nivel={chunk['risk']['level']}")
            for hit in chunk["risk"]["hits"]:
                print(
                    f"   - {hit['category']} (score {hit['category_score']}): {', '.join(hit['terms_found'])}")
    else:
        print(
            "\n[Sinais] Nenhum trecho com sinais relevantes foi encontrado pelo lexico.")

    categories = risk.get("categories_summary", [])
    if categories:
        print("\n[Sinais] Categorias agregadas no áudio:")
        for item in categories:
            print(
                f" - {item['category']} | score_total={item['score_sum']} | chunks={item['chunk_count']} | termos={item['term_mentions']}"
            )

    conclusion = final_summary.get("conclusion", {})
    if conclusion:
        print("\n[Análise]")
        print(f" - Classificacao: {conclusion.get('classification')}")
        print(f" - Leitura: {conclusion.get('message')}")


def main() -> None:
    # Caminhos dos arquivos com base no diretorio do script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Processar o áudio, gerar transcrição, análise de tópicos e risco, e salvar os resultados
    audio_path = os.path.join(script_dir, AUDIO_FILE)
    transcript_txt_path = os.path.join(script_dir, TRANSCRIPT_FILE)
    report_json_path = os.path.join(script_dir, REPORT_FILE)

    # Processar o áudio em chunks, atribuir falas a speakers, gerar resumo por speaker, e construir chunks filtrados para análise
    chunks = transcribe_in_chunks(
        audio_path, language=LANGUAGE, chunk_seconds=CHUNK_SECONDS)
    diarized_turns = assign_speakers_heuristic(chunks, speaker_map=SPEAKER_MAP)
    speaker_summary = summarize_speakers(diarized_turns)

    # Gerar chunks filtrados por papel, mantendo somente as falas do paciente para análise de tópicos e risco
    patient_chunks = build_role_only_chunks(chunks, role=ANALYSIS_ROLE)

    # Gerar um txt com a transcrição completa e timestamps, incluindo os papéis dos falantes
    write_transcript_txt(chunks, transcript_txt_path)
    print(
        f"[Áudio] Transcrição com timestamps salva em: {transcript_txt_path}")

    # Gerar tópicos LDA a partir dos chunks filtrados, análise de risco, resumo de assuntos discutidos, e conclusão geral
    lda_report = lda_topics_from_chunks(
        patient_chunks, num_topics=LDA_TOPICS, passes=LDA_PASSES)
    risk_report = risk_over_chunks(patient_chunks)
    subjects = summarize_discussed_subjects(lda_report, max_topics=4)
    conclusion = build_analysis(patient_chunks, risk_report)

    # Selecionar os chunks com maior score de risco para incluir no relatório
    top_chunks = sorted(
        [c for c in risk_report["by_chunk"] if c["risk"]["score"] > 0],
        key=lambda x: x["risk"]["score"],
        reverse=True,
    )[:5]

    # Gerar o relatório final em formato JSON, incluindo todos os dados relevantes da análise
    report = {
        "audio_file": os.path.basename(audio_path),
        "transcription": {
            "output_txt": os.path.basename(transcript_txt_path),
            "chunks": len(chunks),
            "diarization_mode": speaker_summary["diarization_mode"],
            "speaker_turns": len(diarized_turns),
            "analysis_scope": "patient_speech",
            "patient_chunks_used_in_analysis": len(patient_chunks),
        },
        "lda_topics": lda_report,
        "risk_detection": {
            "nota": "Indicador de engenharia baseado em lexico; nao representa diagnostico clinico.",
            "overall_level": risk_report["overall_level"],
            "avg_score": risk_report["avg_score"],
            "top_chunks": top_chunks,
            "categories_summary": risk_report["categories_summary"],
            "terms_summary": risk_report["terms_summary"],
        },
        "final_summary": {
            "assuntos_discutidos": subjects,
            "conclusion": conclusion,
            "speaker_summary": speaker_summary,
        },
    }

    # Salvar o relatório em formato JSON
    with open(report_json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    # Imprimir o caminho do relatório salvo e exibir um resumo da análise
    print(f"[Áudio] Report JSON salvo em: {report_json_path}")
    get_audio_report(report)


if __name__ == "__main__":
    main()
