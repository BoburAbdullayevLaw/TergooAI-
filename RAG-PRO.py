#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏛️ SEMANTIC TAG-BASED LEGAL RAG ECOSYSTEM
Enterprise-Grade Professional Legal-Tech (2026)

10 ta Strategik Ideya:
1. Tag-Based Semantic Filtering
2. Multi-Step Reasoning
3. Contextual Metadata Injection
4. Self-Correction Loop
5. Dynamic Prompting
6. Recursive Summarization
7. Hybrid Retrieval (Tag + Vector)
8. Entity-Relationship Mapping
9. Citation Generation
10. Uncertainty Quantification
"""

import os
import json
import re
import sys
import logging
from typing import List, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

# LangChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.memory import ConversationBufferWindowMemory
from tqdm import tqdm

# Search (rank-bm25 o'rniga scikit-learn)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.environ["GOOGLE_API_KEY"] = ""
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ============================================================================
# 1. SEMANTIC TAG PARSING (Teglarni ma'no qilib o'qish)
# ============================================================================
@dataclass
class SemanticTag:
    """Semantik teg modeli"""
    tag_type: str  # intent_validation, legal_status_check, reasoning
    value: str  # checked, verified, senior_lawyer
    confidence: float
    is_metadata: bool = True


class TagParser:
    """Teglarni semantiklik asosida tahlil qilish"""

    @staticmethod
    def extract_tags(text: str) -> List[SemanticTag]:
        """Matndan barcha teglarni ajratib olish"""
        tags = []

        # Pattern: <tag_type: value>
        pattern = r'<(\w+):\s*(\w+(?:\s+\w+)?)>'
        matches = re.findall(pattern, text)

        for tag_type, value in matches:
            tag = SemanticTag(
                tag_type=tag_type.lower(),
                value=value.lower(),
                confidence=0.95,
                is_metadata=True
            )
            tags.append(tag)

        return tags

    @staticmethod
    def clean_text_from_tags(text: str) -> str:
        """Matndan teglarni olib tashlash"""
        # Boshlanib turgan teglarni o'chirish
        text = re.sub(r'<bos>', '', text)
        text = re.sub(r'<eos>', '', text)

        # Start/End turn teglarini o'chirish
        text = re.sub(r'<start_of_turn>user\n', '', text)
        text = re.sub(r'<start_of_turn>model\n', '', text)
        text = re.sub(r'</start_of_turn>\n', '\n', text)
        text = re.sub(r'<end_of_turn>\n', '\n', text)

        # Analysis summary tegini o'chirish (lekin ma'lum qilish uchun qayd etish)
        text = re.sub(r'<analysis_summary>.*?</analysis_summary>\n*', '', text, flags=re.DOTALL)

        # Ekstra bo'shliqlarni tozalash
        text = " ".join(text.split())

        return text.strip()


# ============================================================================
# 2. CONTEXTUAL METADATA INJECTION (Metadatani qidiruv indexiga qo'shish)
# ============================================================================
@dataclass
class EnrichedDocument:
    """Teglar bilan boyitilgan hujjat"""
    original_text: str
    clean_text: str
    tags: List[SemanticTag]
    reasoning_level: str  # novice, intermediate, senior_lawyer
    legal_status: str  # unverified, verified, draft
    intent_confidence: float
    modda_number: str
    context: str


class MetadataEnricher:
    """Dokumentlarni teglar asosida boyitish"""

    @staticmethod
    def enrich_document(doc: Document) -> EnrichedDocument:
        """Dokumentni teglar bilan boyitish"""

        text = doc.page_content
        tags = TagParser.extract_tags(text)
        clean_text = TagParser.clean_text_from_tags(text)

        # Tag qiymatlarini aniqlash
        tag_dict = {tag.tag_type: tag.value for tag in tags}

        reasoning_level = tag_dict.get('reasoning', 'novice')
        legal_status = tag_dict.get('legal_status_check', 'unverified')
        intent_validated = tag_dict.get('intent_validation', 'unchecked')
        intent_confidence = 0.95 if intent_validated == 'checked' else 0.5

        modda_number = doc.metadata.get('modda_number', 'N/A')
        context = doc.metadata.get('context', 'General')

        return EnrichedDocument(
            original_text=text,
            clean_text=clean_text,
            tags=tags,
            reasoning_level=reasoning_level,
            legal_status=legal_status,
            intent_confidence=intent_confidence,
            modda_number=modda_number,
            context=context
        )


# ============================================================================
# 3. TAG-BASED SEMANTIC FILTERING (Teglar bo'yicha filtrlash)
# ============================================================================
class SemanticFilterer:
    """Teglar asosida dokumentlarni filtrlash"""

    @staticmethod
    def filter_by_reasoning_level(documents: List[EnrichedDocument], level: str) -> List[EnrichedDocument]:
        """Fikrlash darajasi bo'yicha filtrlash"""
        levels = {'novice': 1, 'intermediate': 2, 'senior_lawyer': 3}
        required_level = levels.get(level, 1)

        filtered = [
            doc for doc in documents
            if levels.get(doc.reasoning_level, 1) >= required_level
        ]

        return sorted(filtered, key=lambda x: levels.get(x.reasoning_level, 1), reverse=True)

    @staticmethod
    def filter_by_verification_status(documents: List[EnrichedDocument], verified_only: bool = True) -> List[
        EnrichedDocument]:
        """Tekshirilgan ma'lumotlar bo'yicha filtrlash"""
        if verified_only:
            return [doc for doc in documents if doc.legal_status == 'verified']
        return documents

    @staticmethod
    def filter_by_confidence(documents: List[EnrichedDocument], min_confidence: float = 0.8) -> List[EnrichedDocument]:
        """Ishonchlilik bo'yicha filtrlash"""
        return [doc for doc in documents if doc.intent_confidence >= min_confidence]


# ============================================================================
# 4. MULTI-STEP REASONING (Ko'p bosqichli fikrlash)
# ============================================================================
class InternalReasoningEngine:
    """AI uchun ichki fikrlash tizimi"""

    def __init__(self, llm):
        self.llm = llm

    def generate_thought_process(self, query: str, documents: List[EnrichedDocument]) -> Dict:
        """Ichki fikrlash jarayonini bosqichma-bosqich yaratish"""

        thought_prompt = f"""Siz senior yuridik tahlilchisiz. Quyidagi savolga chuqur fikrlash zanjirini tuzib bering.

SAVOL: {query}

KONTEKST:
{chr(10).join([f"- {doc.clean_text[:150]}" for doc in documents[:3]])}

ICHKI FIKRLASH BOSQICHLARI:
1. Intent Validation: Savol nima haqida?
2. Relevant Data: Qaysi ma'lumot kerak?
3. Legal Analysis: Huquqiy aspekti qanday?
4. Reasoning Level: Senior lawyer darajasida?
5. Confidence: Ishonch darajasi qancha?

JSON formatda:
{{
    "step_1_intent": "...",
    "step_2_relevance": "...",
    "step_3_analysis": "...",
    "step_4_reasoning": "...",
    "step_5_confidence": 0.0-1.0
}}
"""

        try:
            response = self.llm.invoke(thought_prompt)
            import json
            return json.loads(response.content)
        except Exception as e:
            logger.warning(f"Reasoning generation xatosi: {e}")
            return {"error": "Reasoning generation failed"}


# ============================================================================
# 5. HYBRID TAG + VECTOR RETRIEVAL (Teglar + Vektorli qidiruv)
# ============================================================================
class HybridTagVectorRetriever:
    """Teglar va vektorlar bilan gibrid qidiruv"""

    def __init__(self, enriched_docs: List[EnrichedDocument], vectorstore):
        self.enriched_docs = enriched_docs
        self.vectorstore = vectorstore
        self.corpus = [doc.clean_text for doc in enriched_docs]

        # scikit-learn TF-IDF o'rniga BM25
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            lowercase=True
        )

        try:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.corpus)
        except Exception as e:
            logger.warning(f"TF-IDF initialization xatosi: {e}")
            self.tfidf_matrix = None

    def hybrid_retrieve(self, query: str, k: int = 5, require_verified: bool = True) -> List[
        Tuple[EnrichedDocument, float]]:
        """
        1. Vektorli qidiruv
        2. Tag-based filtrlash
        3. TF-IDF relevance re-ranking
        """

        # 1. VEKTORLI QIDIRUV
        vector_docs = self.vectorstore.similarity_search(query, k=k * 2)

        # 2. TAG-BASED FILTRLASH
        enriched_filtered = []
        for doc in self.enriched_docs:
            if require_verified and doc.legal_status != 'verified':
                continue
            enriched_filtered.append(doc)

        # 3. TF-IDF SCORING
        results = []

        if self.tfidf_matrix is not None:
            try:
                query_tfidf = self.tfidf_vectorizer.transform([query])
                tfidf_scores = cosine_similarity(query_tfidf, self.tfidf_matrix)[0]

                for idx, doc in enumerate(enriched_filtered[:k]):
                    # Tag confidence + TF-IDF score
                    combined_score = doc.intent_confidence * 0.7
                    if idx < len(tfidf_scores):
                        combined_score += tfidf_scores[idx] * 0.3

                    results.append((doc, combined_score))
            except Exception as e:
                # Fallback: faqat tag confidence
                logger.warning(f"TF-IDF scoring xatosi: {e}")
                for doc in enriched_filtered[:k]:
                    results.append((doc, doc.intent_confidence))
        else:
            # Fallback: faqat tag confidence
            for doc in enriched_filtered[:k]:
                results.append((doc, doc.intent_confidence))

        # Skor bo'yicha saralash
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]


# ============================================================================
# 6. ENTITY-RELATIONSHIP MAPPING (Bilimlar grafi)
# ============================================================================
@dataclass
class LegalEntity:
    """Yuridik subyekt"""
    name: str
    entity_type: str  # modda, concept, principle
    modda_number: str
    related_entities: List[str] = None


class KnowledgeGraphBuilder:
    """Bilimlar grafi yaratish"""

    def __init__(self):
        self.entities: Dict[str, LegalEntity] = {}
        self.relationships: List[Tuple[str, str]] = []

    def add_entity(self, entity: LegalEntity):
        """Entity qo'shish"""
        self.entities[entity.name] = entity

    def add_relationship(self, entity1: str, entity2: str):
        """Munosabat qo'shish"""
        self.relationships.append((entity1, entity2))

    def get_related_concepts(self, entity_name: str) -> List[str]:
        """Bog'langan tushunchalarni olish"""
        return [
            target for source, target in self.relationships
            if source == entity_name
        ]


# ============================================================================
# 7. DYNAMIC PROMPTING (Dinamik ko'rsatmalar)
# ============================================================================
class DynamicPromptGenerator:
    """Teglar asosida dinamik ko'rsatmalar"""

    @staticmethod
    def generate_system_prompt(reasoning_level: str, legal_status: str) -> str:
        """Sistemali ko'rsatmani yaratish"""

        base_prompt = "Siz professional yuridik yordamchisiz. "

        # Fikrlash darajasi bo'yicha
        if reasoning_level == 'senior_lawyer':
            base_prompt += "Senior yuridik ekspert sifatida, chuqur tahlil qiling. "
        elif reasoning_level == 'intermediate':
            base_prompt += "O'rta darajadagi tahlil qiling. "
        else:
            base_prompt += "Asosiy ma'lumot bering. "

        # Status bo'yicha
        if legal_status == 'verified':
            base_prompt += "Faqat tekshirilgan ma'lumotni ishlatib javob bering. "
        else:
            base_prompt += "Ehtiyotkorlik bilan javob bering va ishonch darajasini ko'rsating. "

        return base_prompt


# ============================================================================
# 8. SELF-CORRECTION LOOP (O'z-o'zini tekshirish)
# ============================================================================
class SelfCorrectionValidator:
    """Javobning to'g'riligi va mos ligini tekshirish"""

    def __init__(self, llm):
        self.llm = llm

    def validate_against_tags(self, answer: str, enriched_docs: List[EnrichedDocument]) -> Tuple[bool, float, str]:
        """Javobni teglar asosida tekshirish"""

        # Average tag confidence
        avg_confidence = sum(doc.intent_confidence for doc in enriched_docs) / len(
            enriched_docs) if enriched_docs else 0.5

        # Check if answer is too generic
        generic_words = ['mumkin', 'balki', 'shunga o\'xshash', 'deyarli']
        generic_count = sum(1 for word in generic_words if word in answer.lower())

        is_valid = avg_confidence >= 0.7 and generic_count < 3

        if not is_valid:
            corrected = f"[Ehtiyotkorlik ⚠️ ] {answer}"
        else:
            corrected = answer

        return is_valid, avg_confidence, corrected


# ============================================================================
# 9. CITATION GENERATION (Avtomatik manba ko'rsatish)
# ============================================================================
class CitationGenerator:
    """Taglar asosida avtomatik footnotlar"""

    @staticmethod
    def generate_citations(enriched_docs: List[EnrichedDocument]) -> str:
        """Manba ko'rsatishi yaratish"""

        citations = []
        for i, doc in enumerate(enriched_docs, 1):
            reasoning_badge = "✓ Senior" if doc.reasoning_level == 'senior_lawyer' else "• Standard"
            status_badge = "✓ Verified" if doc.legal_status == 'verified' else "⚠ Unverified"

            citation = f"[{i}] {doc.modda_number} ({doc.context}) | {reasoning_badge} | {status_badge}"
            citations.append(citation)

        return "\n".join(citations)


# ============================================================================
# 10. UNCERTAINTY QUANTIFICATION (Ishonchsizlikni o'lchash)
# ============================================================================
class UncertaintyQuantifier:
    """Ishonchsizlik darajasini aniqlash"""

    @staticmethod
    def calculate_uncertainty(documents: List[EnrichedDocument]) -> Dict:
        """Ishonchsizlik metrikalari"""

        if not documents:
            return {"uncertainty": 1.0, "message": "Ma'lumot topilmadi"}

        avg_confidence = sum(doc.intent_confidence for doc in documents) / len(documents)
        verified_count = sum(1 for doc in documents if doc.legal_status == 'verified')

        uncertainty = 1.0 - avg_confidence

        if uncertainty > 0.5:
            level = "Yuqori ⚠️"
        elif uncertainty > 0.3:
            level = "O'rta 📌"
        else:
            level = "Past ✓"

        return {
            "uncertainty": uncertainty,
            "verified_percentage": (verified_count / len(documents)) * 100,
            "confidence": avg_confidence,
            "level": level,
            "message": f"Ishonchlilik darajasi: {int(avg_confidence * 100)}%"
        }


# ============================================================================
# 11. MAIN PROFESSIONAL LEGAL RAG AGENT
# ============================================================================
class ProfessionalTagBasedLegalAgent:
    """Enterprise-grade Tag-Based Legal RAG"""

    def __init__(self, vectorstore):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",
            temperature=0.2,
            max_output_tokens=2048
        )

        self.vectorstore = vectorstore
        self.tag_parser = TagParser()
        self.metadata_enricher = MetadataEnricher()
        self.semantic_filterer = SemanticFilterer()
        self.reasoning_engine = InternalReasoningEngine(self.llm)
        self.citation_gen = CitationGenerator()
        self.uncertainty_quantifier = UncertaintyQuantifier()
        self.self_corrector = SelfCorrectionValidator(self.llm)
        self.dynamic_prompt_gen = DynamicPromptGenerator()

        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            output_key="answer",
            return_messages=True,
            k=10
        )

        self.knowledge_graph = KnowledgeGraphBuilder()
        self.turn_count = 0

    def process_query(self, user_query: str) -> Dict:
        """Kompleks qidiruv va javob berish"""

        self.turn_count += 1
        logger.info(f"\n{'=' * 70}\n🔄 TURN #{self.turn_count}: {user_query[:50]}...\n")

        # 1. VEKTORLI QIDIRUV
        logger.info("1️⃣ Vektorli qidiruv...")
        raw_docs = self.vectorstore.similarity_search(user_query, k=8)

        # 2. TAG EXTRACTION VA ENRICHMENT
        logger.info("2️⃣ Teglarni ajratish va boyitish...")
        enriched_docs = []
        for doc in raw_docs:
            enriched = self.metadata_enricher.enrich_document(doc)
            enriched_docs.append(enriched)

        # 3. TAG-BASED FILTERING
        logger.info("3️⃣ Tekshirilgan ma'lumotlar bo'yicha filtrlash...")
        filtered_docs = self.semantic_filterer.filter_by_verification_status(enriched_docs, verified_only=True)
        if not filtered_docs:
            filtered_docs = enriched_docs[:5]

        # 4. SENIOR LAWYER REASONING
        logger.info("4️⃣ Senior lawyer darajasida...")
        reasoning = self.reasoning_engine.generate_thought_process(user_query, filtered_docs)

        # 5. DYNAMIC PROMPTING
        logger.info("5️⃣ Dinamik ko'rsatma yaratish...")
        primary_doc = filtered_docs[0] if filtered_docs else enriched_docs[0]
        system_prompt = self.dynamic_prompt_gen.generate_system_prompt(
            primary_doc.reasoning_level,
            primary_doc.legal_status
        )

        # 6. ANSWER GENERATION
        logger.info("6️⃣ Javob yaratish...")
        answer_prompt = f"""{system_prompt}

SAVOLUSI: {user_query}

KONTEKST:
{chr(10).join([f"[{doc.modda_number}] {doc.clean_text[:200]}..." for doc in filtered_docs[:3]])}

ICHKI FIKRLASH:
{json.dumps(reasoning, ensure_ascii=False)}

JAVOB:"""

        answer = self.llm.invoke(answer_prompt).content

        # 7. SELF-CORRECTION
        logger.info("7️⃣ O'z-o'zini tekshirish...")
        is_valid, confidence, corrected_answer = self.self_corrector.validate_against_tags(answer, filtered_docs)

        # 8. CITATIONS
        logger.info("8️⃣ Manba-havolalar...")
        citations = self.citation_gen.generate_citations(filtered_docs)

        # 9. UNCERTAINTY QUANTIFICATION
        logger.info("9️⃣ Ishonchlilik darajasi...")
        uncertainty_info = self.uncertainty_quantifier.calculate_uncertainty(filtered_docs)

        final_answer = f"{corrected_answer}\n\n📚 MANBALARI:\n{citations}\n\n{uncertainty_info['message']}"

        # 10. MEMORY UPDATE
        self.memory.save_context(
            {"input": user_query},
            {"output": final_answer}
        )

        return {
            "turn": self.turn_count,
            "query": user_query,
            "answer": final_answer,
            "enriched_docs": len(enriched_docs),
            "filtered_docs": len(filtered_docs),
            "reasoning": reasoning,
            "uncertainty": uncertainty_info,
            "is_valid": is_valid,
            "confidence": confidence
        }


# ============================================================================
# MAIN INTERFEYS
# ============================================================================
def main():
    """Asosiy dastur"""

    logger.info("🏛️ SEMANTIC TAG-BASED PROFESSIONAL LEGAL RAG")
    logger.info("=" * 70)

    # FAISS yuklash
    logger.info("📚 FAISS bazasini yuklash...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    try:
        vectorstore = FAISS.load_local(
            "yuridik_faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
        logger.info("✅ FAISS baza yuklandi")
    except:
        logger.error("❌ FAISS bazasi topilmadi")
        sys.exit(1)

    # Agent yaratish
    agent = ProfessionalTagBasedLegalAgent(vectorstore)

    # Interfeys
    logger.info("\n💬 SUHBAT BOSHLANG\n")

    while True:
        user_input = input("👤 Siz: ").strip()

        if user_input.lower() in ['exit', 'quit', 'chiqish']:
            logger.info("\n👋 Xayr!")
            break

        if not user_input:
            continue

        try:
            result = agent.process_query(user_input)
            print(f"\n🤖 AI:\n{result['answer']}\n")
            print(f"📊 QIYMAT: Filtered {result['filtered_docs']}/{result['enriched_docs']} | "
                  f"Confidence: {result['confidence']:.2%} | "
                  f"Uncertainty: {result['uncertainty']['level']}")
            print("-" * 70 + "\n")

        except Exception as e:
            logger.error(f"❌ Xatolik: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
