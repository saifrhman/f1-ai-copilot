#!/usr/bin/env python3
"""
FIA RAG Agent
Loads FIA regulations from PDFs and provides intelligent answers using LangChain + OpenAI
"""

import os
import glob
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

# LangChain imports (would need to be installed)
try:
    from langchain.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import Qdrant
    from langchain.chat_models import ChatOpenAI
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logging.warning("LangChain not available. Using mock implementation.")

# Mock OpenAI for testing
class MockOpenAI:
    def __init__(self, model_name="gpt-4"):
        self.model_name = model_name
    
    def __call__(self, messages):
        # Mock response for testing
        return type('obj', (object,), {
            'content': f"Mock response for: {messages[-1].content if messages else 'No message'}"
        })


class FIAKnowledgeBase:
    """FIA regulations knowledge base using RAG"""
    
    def __init__(self, fia_docs_path: str = "data/fia_docs", openai_api_key: Optional[str] = None):
        self.fia_docs_path = Path(fia_docs_path)
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.vectorstore = None
        self.qa_chain = None
        self.is_initialized = False
        
        # Initialize if LangChain is available
        if LANGCHAIN_AVAILABLE and self.openai_api_key:
            self._initialize_knowledge_base()
        else:
            logging.warning("Using mock FIA knowledge base")
    
    def _initialize_knowledge_base(self):
        """Initialize the knowledge base with FIA documents"""
        try:
            # Load FIA documents
            documents = self._load_fia_documents()
            
            if not documents:
                logging.warning("No FIA documents found")
                return
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            texts = text_splitter.split_documents(documents)
            
            # Create embeddings and vector store
            embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
            self.vectorstore = Qdrant.from_documents(
                texts, 
                embeddings,
                collection_name="fia_regulations"
            )
            
            # Create QA chain
            llm = ChatOpenAI(
                model_name="gpt-4",
                openai_api_key=self.openai_api_key,
                temperature=0.1
            )
            
            prompt_template = PromptTemplate(
                input_variables=["context", "question"],
                template="""
                You are an expert on FIA Formula 1 regulations. Answer the following question based on the provided context from FIA regulations.
                
                Context: {context}
                
                Question: {question}
                
                Answer the question accurately and cite specific FIA rules when possible. If the information is not in the context, say so.
                """
            )
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}),
                chain_type_kwargs={"prompt": prompt_template}
            )
            
            self.is_initialized = True
            logging.info("FIA knowledge base initialized successfully")
            
        except Exception as e:
            logging.error(f"Failed to initialize FIA knowledge base: {e}")
    
    def _load_fia_documents(self) -> List:
        """Load FIA documents from the specified directory"""
        documents = []
        
        if not self.fia_docs_path.exists():
            logging.warning(f"FIA docs path does not exist: {self.fia_docs_path}")
            return documents
        
        # Find all PDF files
        pdf_files = glob.glob(str(self.fia_docs_path / "*.pdf"))
        
        for pdf_file in pdf_files:
            try:
                loader = PyPDFLoader(pdf_file)
                documents.extend(loader.load())
                logging.info(f"Loaded FIA document: {pdf_file}")
            except Exception as e:
                logging.error(f"Failed to load {pdf_file}: {e}")
        
        return documents
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the FIA knowledge base
        
        Args:
            question: Question about FIA regulations
            
        Returns:
            Dictionary with answer and metadata
        """
        if not self.is_initialized:
            return self._mock_query(question)
        
        try:
            result = self.qa_chain({"query": question})
            return {
                "answer": result["result"],
                "source": "fia_rag_agent",
                "confidence": 0.85,  # Placeholder
                "referenced_rules": self._extract_rules(result["result"])
            }
        except Exception as e:
            logging.error(f"Query failed: {e}")
            return {
                "answer": f"Error querying FIA regulations: {str(e)}",
                "source": "fia_rag_agent",
                "confidence": 0.0,
                "referenced_rules": []
            }
    
    def _mock_query(self, question: str) -> Dict[str, Any]:
        """Mock query for testing without LangChain"""
        mock_answers = {
            "track limits": "Track limits violations are governed by Article 38.3. Drivers must stay within the track boundaries defined by the white lines. Exceeding track limits may result in lap time deletion or penalties.",
            "unsafe release": "Unsafe release penalties are covered under Article 38.4. Pit crews must ensure safe release of cars and avoid impeding other drivers. Penalties range from 5-second time penalties to grid drops.",
            "collision": "Collision penalties are determined by the stewards based on Article 38.1. Factors include intent, severity, and impact on race outcome. Penalties can include time penalties, grid drops, or disqualification.",
            "blocking": "Blocking is regulated under Article 38.2. Drivers must not deliberately impede faster cars. Blocking penalties typically result in 5-10 second time penalties.",
            "drs": "DRS (Drag Reduction System) usage is governed by Article 27.5. DRS can only be used in designated zones when within 1 second of the car ahead.",
            "fuel": "Fuel regulations are covered under Article 30. Teams must use fuel that meets FIA specifications. Fuel samples may be taken for analysis.",
            "tires": "Tire regulations are detailed in Article 24. Teams must use FIA-approved compounds and follow prescribed usage rules."
        }
        
        # Find best matching answer
        best_match = "general"
        for key, answer in mock_answers.items():
            if key.lower() in question.lower():
                best_match = key
                break
        
        return {
            "answer": mock_answers.get(best_match, "I don't have specific information about that FIA regulation. Please consult the official FIA sporting regulations."),
            "source": "fia_rag_agent_mock",
            "confidence": 0.7,
            "referenced_rules": [f"Article {hash(question) % 50}.{hash(question) % 10}"]
        }
    
    def _extract_rules(self, answer: str) -> List[str]:
        """Extract referenced FIA rules from answer"""
        # Simple rule extraction - in production would use more sophisticated parsing
        rules = []
        if "Article" in answer:
            import re
            article_matches = re.findall(r"Article \d+\.?\d*", answer)
            rules.extend(article_matches)
        return rules


# Global FIA knowledge base instance
_fia_kb = None

def get_fia_knowledge_base() -> FIAKnowledgeBase:
    """Get or create FIA knowledge base instance"""
    global _fia_kb
    if _fia_kb is None:
        _fia_kb = FIAKnowledgeBase()
    return _fia_kb

def query_fia_regulations(question: str) -> str:
    """
    Query FIA regulations using RAG system
    
    Args:
        question: Question about FIA regulations
        
    Returns:
        Answer based on FIA regulations
    """
    kb = get_fia_knowledge_base()
    result = kb.query(question)
    return result["answer"]


def query_fia_regulations_detailed(question: str) -> Dict[str, Any]:
    """
    Query FIA regulations with detailed response
    
    Args:
        question: Question about FIA regulations
        
    Returns:
        Detailed response with answer, confidence, and referenced rules
    """
    kb = get_fia_knowledge_base()
    return kb.query(question)


# Example usage and testing
if __name__ == "__main__":
    # Test the FIA RAG agent
    print("🏁 FIA RAG Agent Test")
    print("=" * 50)
    
    test_questions = [
        "What are the penalties for track limits violations?",
        "How is unsafe release penalized?",
        "What are the DRS usage rules?",
        "What are the fuel regulations?",
        "What happens if a driver causes a collision?"
    ]
    
    for question in test_questions:
        print(f"\n❓ Question: {question}")
        answer = query_fia_regulations(question)
        print(f"📋 Answer: {answer}")
        print("-" * 50)
    
    # Test detailed query
    print("\n🔍 Detailed Query Test:")
    detailed_result = query_fia_regulations_detailed("What are the tire regulations?")
    print(f"Answer: {detailed_result['answer']}")
    print(f"Confidence: {detailed_result['confidence']}")
    print(f"Referenced Rules: {detailed_result['referenced_rules']}") 