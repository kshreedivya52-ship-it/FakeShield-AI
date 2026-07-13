import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from app.rag.retriever import NewsRetriever
from typing import Dict, Any, List

load_dotenv()


class RAGPipeline:
    def __init__(self, retriever: NewsRetriever = None):
        self.retriever = retriever or NewsRetriever()
        self.llm = self._setup_llm()

    def _setup_llm(self):
        gemini_key = os.getenv("GEMINI_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        if gemini_key:
            try:
                # Try loading Google Gemini LLM
                from langchain_google_genai import ChatGoogleGenerativeAI
                print("Using Google Gemini API for RAG synthesis.")
                return ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=gemini_key)
            except Exception as e:
                print(f"Error loading ChatGoogleGenerativeAI: {e}. Checking OpenAI...")

        if openai_key:
            try:
                # Try loading OpenAI LLM
                from langchain_community.chat_models import ChatOpenAI
                print("Using OpenAI API for RAG synthesis.")
                return ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_key)
            except Exception as e:
                print(f"Error loading ChatOpenAI: {e}. Falling back to demo mode.")

        print("No API keys found. Running RAG in DEMO mode (Rule-based synthesis).")
        return None

    def generate_explanation(self, query: str) -> Dict[str, Any]:
        """
        Retrieves evidence, passes to LLM (or fallback), and returns answer.
        """
        # Retrieve evidence
        evidence = self.retriever.retrieve_evidence(query, top_k=3)

        if not evidence:
            return {
                "evidence": [],
                "explanation": "No relevant articles or evidence found in the vector database to verify this claim.",
            }

        # Build context from evidence
        context = ""
        for idx, ev in enumerate(evidence):
            context += f"Evidence {idx+1}: Title: {ev['title']} | Verdict in Database: {ev['label']}\nContent Snippet: {ev['content'][:300]}\n\n"

        prompt_template = """
You are FakeShield AI Assistant, an expert fact-checker. 
Analyze the user's claim based on the retrieved evidence below and explain if it is FAKE or REAL.
Provide a clear, detailed, and objective explanation summarizing your reasoning.

User Claim: {query}

Retrieved Evidence:
{context}

Response Format:
Begin your explanation with a clear verdict (e.g., "[VERDICT] FAKE NEWS" or "[VERDICT] REAL NEWS"), followed by a bulleted explanation of why, referencing specific titles from the retrieved evidence. Keep it under 150 words.
"""

        # If LLM is available, run LangChain LLM
        if self.llm is not None:
            try:
                formatted_prompt = prompt_template.format(query=query, context=context)
                response = self.llm.invoke(formatted_prompt)
                explanation_text = response.content
            except Exception as e:
                print(f"LLM execution error: {e}. Falling back to demo response.")
                explanation_text = self._generate_demo_response(query, evidence)
        else:
            explanation_text = self._generate_demo_response(query, evidence)

        return {"evidence": evidence, "explanation": explanation_text}

    def _generate_demo_response(self, query: str, evidence: List[Dict[str, Any]]) -> str:
        # Simple rule-based summary for demo mode
        fake_count = sum(1 for ev in evidence if ev["label"] == "FAKE")
        real_count = sum(1 for ev in evidence if ev["label"] == "REAL")

        verdict = "FAKE NEWS" if fake_count >= real_count else "REAL NEWS"

        explanation = f"[VERDICT] {verdict} (Demo Mode)\n\n"
        explanation += "Based on local evidence analysis:\n"
        explanation += f"- We retrieved {len(evidence)} related article(s) matching your query.\n"

        # Highlight top evidence
        top_ev = evidence[0]
        explanation += f"- The closest match in our database is: '{top_ev['title']}' which is labeled as **{top_ev['label']}**.\n"

        if verdict == "FAKE NEWS":
            explanation += f"- Claims similar to '{query}' are refuted by verified reporting. Mainstream outlets point out inaccuracies in the claim.\n"
        else:
            explanation += f"- The claim matches documented truth and is supported by verified reports like: '{top_ev['title']}'.\n"

        explanation += "\n*(Note: Set GEMINI_API_KEY or OPENAI_API_KEY in a .env file to enable advanced AI generation)*"
        return explanation


if __name__ == "__main__":
    # Quick dry-run check
    try:
        pipeline = RAGPipeline()
        res = pipeline.generate_explanation("NASA discovers aliens")
        print("\nExplanation:")
        print(res["explanation"])
    except Exception as e:
        print("RAG Pipeline test failed:", str(e))
