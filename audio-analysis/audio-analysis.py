# Importação das bibliotecas necessarias
import os
import json
import re
import unicodedata
from typing import List, Dict, Any, Tuple

# Importação de bibliotecas para processamento de audio e texto
import speech_recognition as sr
from pydub import AudioSegment

# Importação de bibliotecas para analise de texto e modelagem de topicos
import nltk
import numpy as np
import gensim
from gensim import corpora
from nltk.corpus import stopwords


# Transcrição de audio para texto usando a API do Google Speech Recognition
def transcribe_audio_to_text(audio_path, text_output_path):
    recognizer = sr.Recognizer()

    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)

        try:
            text = recognizer.recognize_google(audio, language="pt-BR")
            print("Transcrição:", text)

            with open(text_output_path, "w", encoding="utf-8") as file:
                file.write(text)

        except sr.UnknownValueError:
            print("Google Speech Recognition nao conseguiu entender o audio")
        except sr.RequestError as e:
            print(
                f"Erro ao solicitar resultados do servico de reconhecimento de fala do Google; {e}")


# Configuracoes para analise de texto
nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("portuguese"))
TOKEN_RE = re.compile(r"[a-zà-ú]+", re.IGNORECASE)


# Lexico de termos relacionados a violencia, trauma e depressao para analise de risco
RISK_LEXICON = {
    "violence_abuse": {
        "peso": 3,
        "termos": {
            "agressao", "agressão", "violencia", "violência", "abuso", "ameaça", "ameaca",
            "coacao", "coação", "medo", "machucou", "machucar", "controle", "controlar",
            "humilhação", "humilhacao", "chantagem", "assédio", "assedio", "agride", "bate", 
        }
    },
    "trauma_stress": {
        "peso": 2,
        "termos": {
            "trauma", "traumático", "traumatico", "pânico", "panico", "crise", "ansiedade",
            "pesadelo", "gatilho", "flashback", "tremor", "hipervigilancia", "hipervigilância", "vitima"
        }
    },
    "depression_low_mood": {
        "peso": 1,
        "termos": {
            "depressao", "depressão", "triste", "desanimado", "culpa",
            "vazio", "apatia", "choro", "chorar"
        }
    }
}


# Classificacao de nivel de risco com base no score calculado e thresholds definidos
def classify_risk_level(score: float, medium_threshold: float, high_threshold: float) -> str:
    if score >= high_threshold:
        return "alto"
    if score >= medium_threshold:
        return "medio"
    return "baixo"


# Funcao para canonicalizar termos, removendo acentos e normalizando para minusculo
def canonicalize_term(term: str) -> str:
    normalized = unicodedata.normalize("NFD", (term or "").lower())
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


# Conversao de segundos para formato MM:SS para facilitar leitura de timestamps
def seconds_to_mmss(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


# Normalizacao de tokens: limpeza, remocao de stop words e filtragem por tamanho
def normalize_tokens(text: str) -> List[str]:
    tokens = TOKEN_RE.findall(text.lower())
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return tokens


# Split do audio em chunks de N segundos para facilitar a transcrição e analise de audios longos
def split_audio(audio_path: str, chunk_seconds: int = 30) -> List[Tuple[AudioSegment, float, float]]:
    """
    Divide o audio em pedacos de N segundos e retorna:
    [(chunk_audio, start_sec, end_sec), ...]
    """
    audio = AudioSegment.from_file(audio_path)
    total_ms = len(audio)
    chunk_ms = chunk_seconds * 1000

    chunks = []
    for i in range(0, total_ms, chunk_ms):
        start_ms = i
        end_ms = min(i + chunk_ms, total_ms)
        chunk = audio[start_ms:end_ms]
        chunks.append((chunk, start_ms / 1000.0, end_ms / 1000.0))
    return chunks


# Transcrição em chunks para suportar audios longos e gerar timestamps aproximados
def transcribe_in_chunks(audio_path: str, language: str = "pt-BR", chunk_seconds: int = 30) -> List[Dict[str, Any]]:
    """
    Transcreve em chunks para suportar audios longos e gerar timestamps aproximados.
    Retorna uma lista de dicionarios com:
    chunk_index, start_sec, end_sec, text, (opcional) error
    """
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True

    chunks = split_audio(audio_path, chunk_seconds=chunk_seconds)
    total_chunks = len(chunks)
    results = []

    print(f"[Audio] Iniciando transcrição em chunks de {chunk_seconds}s...")

    for idx, (chunk_audio, start_s, end_s) in enumerate(chunks, start=1):
        print(
            f"[Audio] Chunk {idx}/{total_chunks} carregado ({seconds_to_mmss(start_s)} - {seconds_to_mmss(end_s)})")

        tmp_path = f"__tmp_chunk_{idx}.wav"
        chunk_audio.export(tmp_path, format="wav")

        with sr.AudioFile(tmp_path) as source:
            audio_data = recognizer.record(source)

        try:
            text = recognizer.recognize_google(audio_data, language=language)
        except sr.UnknownValueError:
            text = ""
        except sr.RequestError as e:
            results.append({
                "chunk_index": idx,
                "start_sec": start_s,
                "end_sec": end_s,
                "text": "",
                "error": f"RequestError: {str(e)}"
            })
            os.remove(tmp_path)
            print(
                "[Audio] Erro ao acessar o servico de transcrição. Vou salvar o que ja foi processado.")
            break

        results.append({
            "chunk_index": idx,
            "start_sec": start_s,
            "end_sec": end_s,
            "text": text
        })

        status = "ok" if text else "sem fala reconhecida"
        print(f"[Audio] Chunk {idx}/{total_chunks} processado: {status}")

        os.remove(tmp_path)

    print(
        f"[Audio] Transcrição em chunks finalizada. Total de chunks: {len(results)}")
    return results


# Escrita da transcrição em um arquivo de texto
def write_transcript_txt(chunks: List[Dict[str, Any]], out_path: str) -> None:
    """
    Salva transcrição com timestamps aproximados por chunk.
    """
    with open(out_path, "w", encoding="utf-8") as f:
        for c in chunks:
            start = seconds_to_mmss(c.get("start_sec", 0))
            end = seconds_to_mmss(c.get("end_sec", 0))
            text = (c.get("text", "") or "").strip()
            if not text:
                continue
            f.write(f"[{start} - {end}] {text}\n\n")


# Geracao de topicos usando LDA a partir dos chunks transcritos
def lda_topics_from_chunks(chunks: List[Dict[str, Any]], num_topics: int = 4, passes: int = 12) -> Dict[str, Any]:
    texts = []

    for c in chunks:
        t = (c.get("text", "") or "").strip()
        if not t:
            continue
        tokens = normalize_tokens(t)
        if tokens:
            texts.append(tokens)

    if len(texts) < 3:
        return {
            "ok": False,
            "reason": "Texto insuficiente apos transcrição para treinar LDA",
            "topics": []
        }

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(t) for t in texts]

    lda = gensim.models.ldamodel.LdaModel(
        corpus=corpus,
        num_topics=num_topics,
        id2word=dictionary,
        passes=passes,
        random_state=42
    )

    topics = []
    for topic_id in range(num_topics):
        words = lda.show_topic(topic_id, topn=10)
        topics.append({
            "topic_id": topic_id,
            "top_words": [{"word": w, "weight": float(p)} for w, p in words]
        })

    # Distribuição de topicos por chunk valido (apos filtro)
    doc_topics = []
    for i, bow in enumerate(corpus):
        dist = lda.get_document_topics(bow, minimum_probability=0.0)
        dist_sorted = sorted(dist, key=lambda x: x[1], reverse=True)
        dominant_topic, dom_prob = dist_sorted[0]
        doc_topics.append({
            "doc_index": i,
            "dominant_topic": int(dominant_topic),
            "dominant_prob": float(dom_prob),
            "topic_distribution": [{"topic": int(t), "prob": float(p)} for t, p in dist]
        })

    avg = np.zeros(num_topics, dtype=float)
    for dt in doc_topics:
        for item in dt["topic_distribution"]:
            avg[item["topic"]] += item["prob"]
    avg /= max(len(doc_topics), 1)

    overall_dominant = int(np.argmax(avg))

    return {
        "ok": True,
        "topics": topics,
        "doc_topics": doc_topics,
        "overall_dominant_topic": overall_dominant,
        "overall_topic_probs": [{"topic": i, "prob": float(avg[i])} for i in range(num_topics)]
    }


# Calculo de score de risco para um texto com base no lexico definido
def risk_score_for_text(text: str) -> Dict[str, Any]:
    lower = (text or "").lower()
    score = 0
    hits = []

    for category, cfg in RISK_LEXICON.items():
        weight = cfg["peso"]
        terms = cfg["termos"]
        found = []

        for term in terms:
            if term in lower:
                found.append(term)

        if found:
            cat_score = weight * len(set(found))
            score += cat_score
            hits.append({
                "category": category,
                "category_score": int(cat_score),
                "terms_found": sorted(list(set(found)))
            })

    # Thresholds para classificacao de risco (ajustaveis conforme necessidade)
    level = classify_risk_level(score, medium_threshold=4, high_threshold=8)

    return {"score": int(score), "level": level, "hits": hits}


# Calculo de risco ao longo dos chunks para identificar trechos com sinais sensiveis
def risk_over_chunks(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    per_chunk = []
    total_score = 0
    category_stats: Dict[str, Dict[str, Any]] = {}
    term_stats: Dict[str, Dict[str, Any]] = {}

    for c in chunks:
        text = c.get("text", "")
        rs = risk_score_for_text(text)
        total_score += rs["score"]
        chunk_index = c.get("chunk_index")

        for hit in rs["hits"]:
            category = hit["category"]
            if category not in category_stats:
                category_stats[category] = {
                    "category": category,
                    "score_sum": 0,
                    "term_mentions": 0,
                    "chunk_indices": set()
                }

            category_stats[category]["score_sum"] += hit["category_score"]
            category_stats[category]["term_mentions"] += len(
                hit["terms_found"])
            if chunk_index is not None:
                category_stats[category]["chunk_indices"].add(chunk_index)

            for term in hit["terms_found"]:
                canonical = canonicalize_term(term)
                if canonical not in term_stats:
                    term_stats[canonical] = {
                        "canonical_term": canonical,
                        "display_term": term,
                        "category": category,
                        "mentions": 0,
                        "chunk_indices": set()
                    }
                term_stats[canonical]["mentions"] += 1
                if chunk_index is not None:
                    term_stats[canonical]["chunk_indices"].add(chunk_index)

        per_chunk.append({
            "chunk_index": chunk_index,
            "start": seconds_to_mmss(c.get("start_sec", 0)),
            "end": seconds_to_mmss(c.get("end_sec", 0)),
            "risk": rs
        })

    avg_score = total_score / max(len(chunks), 1)
    overall = classify_risk_level(
        avg_score, medium_threshold=1.0, high_threshold=2.0)

    categories_summary = []
    for item in category_stats.values():
        chunk_indices = sorted(list(item["chunk_indices"]))
        categories_summary.append({
            "category": item["category"],
            "score_sum": int(item["score_sum"]),
            "term_mentions": int(item["term_mentions"]),
            "chunk_count": len(chunk_indices),
            "chunks": chunk_indices
        })

    categories_summary = sorted(
        categories_summary,
        key=lambda x: (-x["score_sum"], -x["chunk_count"], x["category"])
    )

    terms_summary = []
    for item in term_stats.values():
        chunk_indices = sorted(list(item["chunk_indices"]))
        terms_summary.append({
            "term": item["display_term"],
            "canonical_term": item["canonical_term"],
            "category": item["category"],
            "mentions": int(item["mentions"]),
            "chunk_count": len(chunk_indices),
            "chunks": chunk_indices
        })

    terms_summary = sorted(
        terms_summary,
        key=lambda x: (-x["chunk_count"], -x["mentions"], x["canonical_term"])
    )

    return {
        "overall_level": overall,
        "avg_score": round(avg_score, 2),
        "by_chunk": per_chunk,
        "categories_summary": categories_summary,
        "terms_summary": terms_summary
    }


# Resumo dos assuntos discutidos com base nos topicos gerados pelo LDA e suas palavras-chave
def summarize_discussed_subjects(lda_report: Dict[str, Any], max_topics: int = 4) -> List[Dict[str, Any]]:
    if not lda_report.get("ok"):
        return []

    probs = sorted(
        lda_report.get("overall_topic_probs", []),
        key=lambda x: x.get("prob", 0.0),
        reverse=True
    )[:max_topics]

    subjects = []
    for item in probs:
        topic_id = item["topic"]
        topic_def = next(
            (t for t in lda_report.get("topics", [])
             if t.get("topic_id") == topic_id),
            None
        )
        if not topic_def:
            continue

        keywords = [w["word"] for w in topic_def.get("top_words", [])[:8]]
        subjects.append({
            "topic_id": topic_id,
            "presence": round(float(item.get("prob", 0.0)), 3),
            "keywords": keywords
        })

    return subjects


# Analise final do audio combinando os resultados de risco e topicos para classificação
def build_analysis(chunks: List[Dict[str, Any]], risk_report: Dict[str, Any]) -> Dict[str, Any]:
    categories = {c["category"]: c for c in risk_report.get(
        "categories_summary", [])}
    terms = {t["canonical_term"]
        : t for t in risk_report.get("terms_summary", [])}

    has_violence_abuse = "violence_abuse" in categories
    has_trauma = "trauma_stress" in categories
    has_depression = "depression_low_mood" in categories

    violence_mentions = terms.get("violencia", {}).get("chunk_count", 0)
    total_chunks = max(len(chunks), 1)
    sensitive_chunks = len(
        [c for c in risk_report.get("by_chunk", []) if c.get(
            "risk", {}).get("score", 0) > 0]
    )
    sensitive_coverage = round(sensitive_chunks / total_chunks, 3)
    overall_level = risk_report.get("overall_level")
    avg_score = risk_report.get("avg_score")

    dominant_category = None
    if categories:
        dominant_category = sorted(
            categories.values(),
            key=lambda x: (x.get("score_sum", 0), x.get("chunk_count", 0)),
            reverse=True
        )[0]["category"]

    repeated_violence_signal = has_violence_abuse and (
        violence_mentions >= 2 or categories["violence_abuse"]["chunk_count"] >= 2
    )
    strong_sensitive_signal = overall_level in {
        "medio", "alto"} or sensitive_coverage >= 0.25

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
        "non_clinical_notice": "Resultado baseado em regras lexicais; nao representa diagnostico clinico.",
        "evidence": {
            "overall_level": overall_level,
            "avg_score": avg_score,
            "dominant_sensitive_category": dominant_category,
            "sensitive_chunks": sensitive_chunks,
            "total_chunks": total_chunks,
            "sensitive_chunk_coverage": sensitive_coverage,
            "violence_mentions_in_chunks": violence_mentions,
            "categories_detected": list(categories.keys()),
            "top_sensitive_terms": [t["term"] for t in risk_report.get("terms_summary", [])[:10]]
        }
    }


# Impressão de insights
def get_audio_report(report: Dict[str, Any]) -> None:
    print("\nANÁLISE DO AUDIO")
    print("_______________________________")

    lda = report.get("lda_topics", {})
    risk = report.get("risk_detection", {})
    final_summary = report.get("final_summary", {})

    # Topicos
    if lda.get("ok"):
        dom = lda.get("overall_dominant_topic")
        print(f"\n[Topicos] Topico dominante geral: {dom}")

        probs = sorted(lda.get("overall_topic_probs", []),
                       key=lambda x: x["topic"])
        for item in probs:
            tid = item["topic"]
            prob = item["prob"]
            topic_def = next(
                (t for t in lda["topics"] if t["topic_id"] == tid), None)
            if topic_def:
                palavras = ", ".join([w["word"]
                                     for w in topic_def["top_words"][:8]])
                print(
                    f" - Topico {tid} | presenca ~ {prob:.2f} | palavras: {palavras}")
    else:
        print("\n[Topicos] Nao foi possivel gerar topicos (texto insuficiente).")

    # Risco
    print("\n[Sinais] Indicador (nao clinico) de temas sensiveis:")
    print(f" - Nivel geral: {risk.get('overall_level')}")
    print(f" - Score medio: {risk.get('avg_score')}")

    top_chunks = risk.get("top_chunks", [])
    if top_chunks:
        print("\n[Sinais] Trechos com maior evidencia (top):")
        for c in top_chunks:
            start = c["start"]
            end = c["end"]
            score = c["risk"]["score"]
            level = c["risk"]["level"]
            print(f" - [{start} - {end}] score={score} nivel={level}")
            for hit in c["risk"]["hits"]:
                termos = ", ".join(hit["terms_found"])
                print(
                    f"   - {hit['category']} (score {hit['category_score']}): {termos}")
    else:
        print(
            "\n[Sinais] Nenhum trecho com sinais relevantes foi encontrado pelo lexico.")

    cat_summary = risk.get("categories_summary", [])
    if cat_summary:
        print("\n[Sinais] Categorias agregadas no audio:")
        for item in cat_summary:
            print(
                f" - {item['category']} | score_total={item['score_sum']} | chunks={item['chunk_count']} | termos={item['term_mentions']}")

    conclusion = final_summary.get("conclusion", {})
    if conclusion:
        print("\n[Conclusao]")
        print(f" - Classificacao: {conclusion.get('classification')}")
        print(f" - Leitura: {conclusion.get('message')}")


# Funcao principal para execucao do script
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    audio_path = os.path.join(script_dir, "audio-analysis.wav")
    transcript_txt_path = os.path.join(script_dir, "transcricao-audio.txt")
    report_json_path = os.path.join(script_dir, "relatorio_audio.json")

    # 1. Transcrição em chunks
    chunks = transcribe_in_chunks(
        audio_path, language="pt-BR", chunk_seconds=30)

    # 2. Salva transcrição com timestamps
    write_transcript_txt(chunks, transcript_txt_path)
    print(
        f"[Audio] Transcrição com timestamps salva em: {transcript_txt_path}")

    # 3. Gera topicos (LDA) e sinais (lexico)
    lda_report = lda_topics_from_chunks(chunks, num_topics=4, passes=12)
    risk_report = risk_over_chunks(chunks)
    subjects = summarize_discussed_subjects(lda_report, max_topics=4)
    conclusion = build_analysis(chunks, risk_report)

    top_chunks = sorted(
        [c for c in risk_report["by_chunk"] if c["risk"]["score"] > 0],
        key=lambda x: x["risk"]["score"],
        reverse=True
    )[:5]

    report = {
        "audio_file": os.path.basename(audio_path),
        "transcription": {
            "output_txt": os.path.basename(transcript_txt_path),
            "chunks": len(chunks)
        },
        "lda_topics": lda_report,
        "risk_detection": {
            "nota": "Indicador de engenharia baseado em lexico; nao representa diagnostico clinico.",
            "overall_level": risk_report["overall_level"],
            "avg_score": risk_report["avg_score"],
            "top_chunks": top_chunks,
            "categories_summary": risk_report["categories_summary"],
            "terms_summary": risk_report["terms_summary"]
        },
        "final_summary": {
            "assuntos_discutidos": subjects,
            "conclusion": conclusion
        },
    }

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[Audio] Report JSON salvo em: {report_json_path}")

    # 4. Imprime insights do audio
    get_audio_report(report)


if __name__ == "__main__":
    main()
