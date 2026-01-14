import json
import os
import sys
import time
from typing import List, Dict, Tuple
from pathlib import Path
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from tqdm import tqdm
import logging

# ============================================================================
# 1. LOGGING KONFIGURATSIYASI
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# 2. API SOZLAMALARI
# ============================================================================
os.environ["GOOGLE_API_KEY"] = ""
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ============================================================================
# 3. SOZLAMALAR VA KONSTANTALAR
# ============================================================================
CONFIG = {
    "INPUT_FILE": "b01_uz_fixed.jsonl",
    "OUTPUT_DIR": "yuridik_faiss_index",
    "EMBEDDING_MODEL": "models/text-embedding-004",
    "BATCH_SIZE": 100,  # Batch bilan ishish
    "MIN_CHUNK_SIZE": 150,  # Minimal token hajmi
    "MAX_CHUNK_SIZE": 1500,  # Maksimal token hajmi
    "CHUNK_OVERLAP": 50,  # Chunklarning uyg'unligi
    "VERBOSE": True
}


# ============================================================================
# 4. MATN QAYTA ISHLAB CHIQARISH (CLEANING & CHUNKING)
# ============================================================================
class TextProcessor:
    """
    Matnni qidirish uchun optimal darajada qayta ishlab chiqaradi.
    """

    @staticmethod
    def clean_text(text: str) -> str:
        """Matnni tozalash va normallashtirish"""
        if not text:
            return ""

        # Ekstra bo'shliqlarni olib tashlash
        text = " ".join(text.split())

        # HTML va markdown taglari olib tashlash
        text = text.replace("<bos>", "").replace("<eos>", "")
        text = text.replace("<start_of_turn>", " ").replace("</start_of_turn>", " ")
        text = text.replace("user", "").replace("model", "").replace("analysis_summary", "")

        # Takrorlangan belgilerni o'chirish
        text = text.replace("***", "").replace("***", "")

        return text.strip()

    @staticmethod
    def smart_chunk(text: str, max_chunk: int = 1500, overlap: int = 50) -> List[str]:
        """
        Matnni intelligent chunklarga bo'lish (gaplar bo'yicha)
        """
        if len(text) <= max_chunk:
            return [text]

        chunks = []
        sentences = text.replace("! ", "!|").replace("? ", "?|").replace(". ", ".|").split("|")

        current_chunk = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(current_chunk) + len(sentence) <= max_chunk:
                current_chunk += " " + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence[-overlap:] + " " + sentence if overlap > 0 else sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return [c for c in chunks if len(c) > 50]  # Juda kichik chunklarni o'chirish


# ============================================================================
# 5. JSONL READER (XATOSIZ VA SAMARALI)
# ============================================================================
class JSONLReader:
    """
    JSONL faylni xatosiz o'qish va qayta ishlash
    """

    @staticmethod
    def read_and_process(file_path: str) -> Tuple[List[Document], Dict]:
        """
        JSONL faylni o'qib, Document-larga aylantirish
        """
        documents = []
        stats = {
            "total_lines": 0,
            "valid_records": 0,
            "skipped_records": 0,
            "total_chunks": 0,
            "avg_chunk_size": 0
        }

        if not Path(file_path).exists():
            logger.error(f"❌ Fayl topilmadi: {file_path}")
            sys.exit(1)

        processor = TextProcessor()
        chunk_sizes = []

        logger.info(f"📖 JSONL fayldan o'qilmoqda: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(tqdm(f, desc="📚 Qayta ishlash", unit="qator"), 1):
                stats["total_lines"] = line_num

                try:
                    data = json.loads(line)

                    # Matn va context olish
                    text = data.get('text', '').strip()
                    context = data.get('context', 'Umumiy').strip()

                    if not text or len(text) < CONFIG["MIN_CHUNK_SIZE"]:
                        stats["skipped_records"] += 1
                        continue

                    # Matnni tozalash
                    clean_text = processor.clean_text(text)

                    if not clean_text:
                        stats["skipped_records"] += 1
                        continue

                    # Matnni intelligent chunklarga bo'lish
                    chunks = processor.smart_chunk(
                        clean_text,
                        max_chunk=CONFIG["MAX_CHUNK_SIZE"],
                        overlap=CONFIG["CHUNK_OVERLAP"]
                    )

                    # Har bir chunk uchun Document yaratish
                    for chunk_idx, chunk in enumerate(chunks):
                        if len(chunk) >= CONFIG["MIN_CHUNK_SIZE"]:
                            doc = Document(
                                page_content=chunk,
                                metadata={
                                    "source": "yuridik_baza",
                                    "context": context,
                                    "line_id": line_num,
                                    "chunk_id": chunk_idx,
                                    "original_length": len(text)
                                }
                            )
                            documents.append(doc)
                            stats["total_chunks"] += 1
                            chunk_sizes.append(len(chunk))

                    stats["valid_records"] += 1

                except json.JSONDecodeError as e:
                    logger.warning(f"⚠️ JSON xatosi (qator {line_num}): {e}")
                    stats["skipped_records"] += 1
                    continue
                except Exception as e:
                    logger.warning(f"⚠️ Xatolik (qator {line_num}): {e}")
                    stats["skipped_records"] += 1
                    continue

        # Statistika hisoblash
        if chunk_sizes:
            stats["avg_chunk_size"] = sum(chunk_sizes) // len(chunk_sizes)

        return documents, stats


# ============================================================================
# 6. FAISS VEKTORLI BAZA YARATISH (BATCH MODE)
# ============================================================================
class FAISSBuilder:
    """
    FAISS vektorli bazasini batch rejimida yaratish
    """

    @staticmethod
    def build_and_save(documents: List[Document], output_dir: str) -> bool:
        """
        Batch rejimida FAISS bazasini yaratish va saqlash
        """
        if not documents:
            logger.error("❌ Dokumentlar topilmadi!")
            return False

        try:
            logger.info(f"🔐 Embedding modelini yuklash...")
            embeddings = GoogleGenerativeAIEmbeddings(
                model=CONFIG["EMBEDDING_MODEL"]
            )

            logger.info(f"🚀 {len(documents)} ta dokumentdan FAISS baza yaratilmoqda...")
            logger.info(f"   (Batch hajmi: {CONFIG['BATCH_SIZE']})")

            # Dokumentlarni batch-larga bo'lish
            vectorstore = None
            for batch_start in tqdm(
                    range(0, len(documents), CONFIG["BATCH_SIZE"]),
                    desc="🔄 Embedding batch-lash",
                    unit="batch"
            ):
                batch_end = min(batch_start + CONFIG["BATCH_SIZE"], len(documents))
                batch_docs = documents[batch_start:batch_end]

                if vectorstore is None:
                    # Birinchi batch
                    vectorstore = FAISS.from_documents(batch_docs, embeddings)
                else:
                    # Keyingi batch-larni qo'shish
                    batch_vectorstore = FAISS.from_documents(batch_docs, embeddings)
                    vectorstore.merge_from(batch_vectorstore)

            logger.info(f"💾 FAISS baza saqlanmoqda: {output_dir}")
            vectorstore.save_local(output_dir)

            logger.info(f"✅ FAISS baza muvaffaqiyatli yaratildi!")
            return True

        except Exception as e:
            logger.error(f"❌ FAISS yaratishda xatolik: {e}")
            import traceback
            traceback.print_exc()
            return False


# ============================================================================
# 7. QIDIRUV SAMARADORLIGINI TESTING
# ============================================================================
class SearchQualityTester:
    """
    FAISS bazasining qidiruv samaradorligini sinab ko'rish
    """

    @staticmethod
    def test_search(vectorstore, test_queries: List[str]) -> Dict:
        """
        Test qidiruv so'rovlari bilan samaradorlikni tekshirish
        """
        logger.info("\n🔍 Qidiruv samaradorligini sinab ko'rilmoqda...")

        results = {
            "total_queries": len(test_queries),
            "avg_relevance": 0,
            "details": []
        }

        test_queries = test_queries or [
            "jinoyat protsessi",
            "fuqaroviy javobgarlik",
            "yarashuv instituti",
            "sud hukmini",
            "adliya tizimi"
        ]

        for query in test_queries:
            try:
                docs = vectorstore.similarity_search(query, k=3)
                results["details"].append({
                    "query": query,
                    "results_found": len(docs),
                    "top_relevance": docs[0].page_content[:100] if docs else "Yo'q"
                })
                logger.info(f"   ✓ '{query}' -> {len(docs)} ta natija")
            except Exception as e:
                logger.warning(f"   ✗ '{query}' -> Xatolik: {e}")

        return results


# ============================================================================
# 8. MAIN FUNKSIYA
# ============================================================================
def main():
    """
    Asosiy dastur mantiqiy alg'oritmi
    """
    logger.info("=" * 70)
    logger.info("🏛️  O'ZBEKISTON YURIDIK BAZASI - FAISS VEKTORLASH TIZIMI")
    logger.info("=" * 70)

    start_time = time.time()

    # 1. JSONL faylni o'qish va qayta ishlash
    logger.info("\n📖 QADAM 1: JSONL FAYLNI O'QISH")
    documents, stats = JSONLReader.read_and_process(CONFIG["INPUT_FILE"])

    logger.info(f"\n📊 STATISTIKA:")
    logger.info(f"   • Jami qatorlar: {stats['total_lines']}")
    logger.info(f"   • Haqiqiy yozuvlar: {stats['valid_records']}")
    logger.info(f"   • O'tkazib yuborilgan: {stats['skipped_records']}")
    logger.info(f"   • Chunklashtirilgan: {stats['total_chunks']}")
    logger.info(f"   • O'rtacha chunk hajmi: {stats['avg_chunk_size']} belgi")

    if not documents:
        logger.error("❌ Dokumentlar qayta ishlashda xatolik!")
        sys.exit(1)

    # 2. FAISS bazasini yaratish
    logger.info("\n🔧 QADAM 2: FAISS VEKTORLI BAZASINI YARATISH")
    success = FAISSBuilder.build_and_save(documents, CONFIG["OUTPUT_DIR"])

    if not success:
        sys.exit(1)

    # 3. Qidiruv testlari
    logger.info("\n🧪 QADAM 3: QIDIRUV SAMARADORLIGINI SINAB KO'RISH")
    embeddings = GoogleGenerativeAIEmbeddings(
        model=CONFIG["EMBEDDING_MODEL"]
    )
    vectorstore = FAISS.load_local(
        CONFIG["OUTPUT_DIR"],
        embeddings,
        allow_dangerous_deserialization=True
    )

    test_results = SearchQualityTester.test_search(vectorstore, [])

    # 4. Yakuniy hisobot
    elapsed_time = time.time() - start_time
    logger.info("\n" + "=" * 70)
    logger.info("✅ YAKUNIY HISOBOT")
    logger.info("=" * 70)
    logger.info(f"⏱️  Jami vaqt: {elapsed_time:.2f} soniya")
    logger.info(f"📈 Dokumentlar: {len(documents)}")
    logger.info(f"📂 Bazalar joylashgan: {CONFIG['OUTPUT_DIR']}/")
    logger.info(f"🔍 Qidiruv testlari: {test_results['total_queries']} ta")
    logger.info("=" * 70 + "\n")


if __name__ == "__main__":
    main()
