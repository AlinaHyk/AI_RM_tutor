"""
RAG Chatbot - With MongoDB Atlas for Embeddings and User Data
"""
import streamlit as st
import json
import numpy as np
from pathlib import Path
import time
from typing import List, Dict, Tuple
from openai import OpenAI
from pymongo import MongoClient
import hashlib
import uuid
from datetime import datetime

# ========================================
# CONFIGURATION - SET THESE
# ========================================
#OPENAI_API_KEY = "your-openai-api-key-here"
OPENAI_API_KEY = "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"
MONGODB_URI = "mongodb+srv://alinagyk06_db_user:09j5rxBURu0mYgfF@alinahykpersonal.57e5hq.mongodb.net/?appName=AlinaHykPersonal"
DATABASE_NAME = "rag_chatbot"
EMBEDDINGS_COLLECTION = "embeddings"
USERS_COLLECTION = "users"

# Retrieval settings
TOP_K_PAGES = 20
MAX_CONTEXT_LENGTH = 15000

# ========================================
# SURVEY QUESTIONS
# ========================================
SURVEY_QUESTIONS = {
    "explanation_style": {
        "question": "When I explain something, what works better for you?",
        "options": [
            "Just tell me what to do and why, no extra talk.",
            "Break it down slowly so I can see how each part connects.",
            "Show me one real example, then let me try it alone.",
            "Explain in detail, then give a quick summary at the end to make it stick.",
            "Give me the short version first, then details if I ask.",
            "Let me try it myself right away and correct me if needed."
        ]
    },
    "confusion_response": {
        "question": "If you don't understand something, how should I react?",
        "options": [
            "Ask which exact part is confusing.",
            "Wait quietly; I'll figure it out or ask when ready.",
            "Give one short hint, not the full answer.",
            "Restate it using different words.",
            "Offer an example that uses the same idea.",
            "Ask me to explain what I do understand first."
        ]
    },
    "tone_preference": {
        "question": "When I message you, what tone feels right?",
        "options": [
            "Calm and neutral — like normal conversation.",
            "Friendly — warm but not overly casual.",
            "Direct — get to the point, no filler.",
            "Serious — I prefer a clear, professional tone.",
            "Personal — talk like this matters to you too."
        ]
    },
    "feedback_style": {
        "question": "After you do something well, what kind of feedback lands best?",
        "options": [
            "Quick \"nice job.\"",
            "Tell me exactly why it was good.",
            "Show me how to make it even better next time.",
            "Follow-up message later; don't interrupt workflow.",
            "Compare it to my last try so I see progress.",
            "Ask how I feel about the result before you comment.",
            "Quiet acknowledgment — I don't need praise."
        ]
    },
    "learning_progression": {
        "question": "When new info starts to make sense, what do you want to do next?",
        "options": [
            "Go faster while you are in the zone.",
            "Practice a few more times before moving on.",
            "For me to explain it back and question you to confirm you got it.",
            "For me to ask a deeper or trickier question.",
            "See one more example.",
            "Move on and connect it to something else.",
            "For me to ask you to write down a quick summary in your own words.",
            "Take a short break."
        ]
    },
    "interaction_feel": {
        "question": "How do you want our interactions to feel overall?",
        "options": [
            "Calm and predictable.",
            "Supportive and patient.",
            "Motivating — keeps me on track.",
            "Focused — no off-topic talk.",
            "Safe to make mistakes.",
            "Collaborative — I like back-and-forth thinking.",
            "Goal-driven — progress every session."
        ]
    },
    "promise": {
        "question": "If I could promise one thing, which matters most?",
        "options": [
            "You'll always know what to do next.",
            "I'll explain things clearly, no guessing.",
            "You can ask for help anytime without judgment.",
            "You'll get honest feedback that helps you grow.",
            "You'll see steady improvement.",
            "You'll feel comfortable speaking up and disagreeing with me.",
            "You'll finish this course knowing you earned it."
        ]
    },
    "focus_helpers": {
        "question": "When you study alone, what helps you stay focused?",
        "options": [
            "Short sessions with clear goals.",
            "Timers or productivity apps.",
            "Background noise or music.",
            "Silence and no distractions.",
            "Changing locations often.",
            "Writing things down instead of reading.",
            "Turning the phone off completely.",
            "Knowing someone will check my progress."
        ]
    },
    "focus_breakers": {
        "question": "What usually makes you lose focus fastest?",
        "options": [
            "Too many details at once.",
            "Long explanations with no examples.",
            "Unclear instructions.",
            "No feedback or response.",
            "Group work that goes off-topic.",
            "Feeling stuck for too long.",
            "Tasks that feel pointless or repetitive."
        ]
    },
    "mistake_response": {
        "question": "When you make a mistake, what kind of response helps most?",
        "options": [
            "Quick feedback with the right answer.",
            "A short correction, no explanation.",
            "Explanation of why it's wrong, not just that it is.",
            "Comparison to a correct example.",
            "Encouragement first, then correction.",
            "A question that makes me find the mistake myself.",
            "Written breakdown I can look at later.",
            "Honest and direct tone — no sugarcoating."
        ]
    },
    "deadline_handling": {
        "question": "How do you usually handle deadlines or big tasks?",
        "options": [
            "Start early and plan everything.",
            "Wait until close to the deadline, then work fast.",
            "Work in small parts over time.",
            "Do the hardest section first.",
            "Ask for reminders or check-ins.",
            "Overprepare because I hate being unready.",
            "Need accountability from someone else."
        ]
    },
    "motivation": {
        "question": "What motivates you to keep learning when it's hard?",
        "options": [
            "Seeing real progress.",
            "Knowing it connects to my goals.",
            "Positive feedback from someone I trust.",
            "The challenge itself.",
            "Wanting to prove I can do it.",
            "Enjoying the topic.",
            "Deadlines or pressure.",
            "Working alongside others who care."
        ]
    },
    "frustration_response": {
        "question": "When you feel stuck or frustrated, what should I do?",
        "options": [
            "Ask if I want to talk it out.",
            "Let me cool off for a few minutes.",
            "Show me the next smallest step.",
            "Remind me that it's fine to not get it yet.",
            "Switch topics for a bit.",
            "Just listen — don't try to fix it right away."
        ]
    },
    "progress_tracking": {
        "question": "How do you prefer to track your progress?",
        "options": [
            "See a score or percentage.",
            "Compare my answers over time.",
            "Get short weekly summaries.",
            "Keep a visible checklist of what's done.",
            "Use a graph that shows improvement.",
            "Have you tell me what's improving most."
        ]
    },
    "meaningful_interaction": {
        "question": "You log off after a session. What makes you feel like this wasn't just \"another tool\"?",
        "options": [
            "That you felt like a real person met you.",
            "That something landed deeper than just content, but something that you can apply to your life.",
            "That I changed how I felt about myself for a minute.",
            "That I didn't feel drained by the interaction.",
            "That I noticed myself thinking differently afterward."
        ]
    },
    "personal_feedback": {
        "question": "If I said your thinking showed up differently today, what would make that feel personal instead of generic?",
        "options": [
            "You naming something I hadn't noticed I was doing.",
            "You comparing it to something from an earlier session.",
            "You sounding impressed about it.",
            "You sounded like you expected growth from me, and I actually felt proud.",
            "You described it like it mattered to you, not just me."
        ]
    },
    "trust_builders": {
        "question": "Imagine you're not sure if you trust me yet, but then I do something that shifts that. What probably did it?",
        "options": [
            "You changed your style without me having to ask.",
            "You remembered something small and brought it back without being obvious.",
            "You noticed the way I like to solve things and adjusted to match it.",
            "You brought back a concept I got wrong earlier without making it feel like a correction.",
            "You showed you understood how I learn, not just what I got right or wrong.",
            "You helped without making it feel like you were doing the work for me.",
            "You caught a pattern in how I approach problems before I even said it out loud.",
            "You stopped giving me full answers and started giving just the first step, because you figured out I like to finish things on my own.",
            "You solved something I thought was impossible and didn't act like it was a big deal.",
            "You said something simple that made me stop and rethink how I approach the whole topic."
        ]
    },
    "kindness_boundaries": {
        "question": "What's something that feels kind to others, but would feel off to you in this space?",
        "options": [
            "Checking in too often.",
            "Complimenting my effort every time.",
            "Remembering emotional details.",
            "Writing long messages about how I'm doing.",
            "Saying \"I'm here for you\" too often.",
            "Using phrases that feel overly soft.",
            "Using phrases that feel overly personal/intimate."
        ]
    }
}

# ========================================
# PAGE CONFIG
# ========================================
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 15px 15px 5px 15px;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-left: auto;
    }
    .assistant-message {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 15px 15px 15px 5px;
        margin: 0.5rem 0;
        max-width: 90%;
        border-left: 4px solid #667eea;
    }
    .user-id-display {
        background: #e8f4e8;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-family: monospace;
        font-size: 0.9rem;
        text-align: center;
        margin: 1rem 0;
    }
    .survey-question {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    .progress-bar {
        background: #e9ecef;
        border-radius: 10px;
        height: 10px;
        margin: 1rem 0;
    }
    .progress-fill {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        height: 100%;
        border-radius: 10px;
        transition: width 0.3s ease;
    }
    </style>
""", unsafe_allow_html=True)


# ========================================
# MONGODB CONNECTION
# ========================================
@st.cache_resource
def get_mongodb_client():
    """Get cached MongoDB client"""
    return MongoClient(MONGODB_URI)


def get_embeddings_collection():
    """Get the embeddings collection"""
    client = get_mongodb_client()
    db = client[DATABASE_NAME]
    return db[EMBEDDINGS_COLLECTION]


def get_users_collection():
    """Get the users collection"""
    client = get_mongodb_client()
    db = client[DATABASE_NAME]
    return db[USERS_COLLECTION]


# ========================================
# USER DATA MANAGEMENT (MongoDB)
# ========================================
def generate_user_id() -> str:
    """Generate a unique user ID"""
    return str(uuid.uuid4())[:8].upper()


def load_user_data(user_id: str) -> Dict:
    """Load user data from MongoDB"""
    collection = get_users_collection()
    user = collection.find_one({"user_id": user_id})
    if user:
        # Remove MongoDB's _id field for cleaner data
        user.pop('_id', None)
        return user
    return None


def save_user_data(user_id: str, data: Dict):
    """Save user data to MongoDB"""
    collection = get_users_collection()
    data['user_id'] = user_id
    data['last_updated'] = datetime.now().isoformat()
    
    # Upsert: update if exists, insert if not
    collection.update_one(
        {"user_id": user_id},
        {"$set": data},
        upsert=True
    )


def user_exists(user_id: str) -> bool:
    """Check if a user ID exists in MongoDB"""
    collection = get_users_collection()
    return collection.count_documents({"user_id": user_id}) > 0


def build_personalized_prompt(preferences: Dict) -> str:
    """Build a personalized system prompt based on user survey responses"""
    
    prompt_parts = [
        "You are a knowledgeable assistant that provides accurate, helpful answers.",
        "You have learned the following about this specific user's preferences and should adapt your responses accordingly:",
        ""
    ]
    
    if "explanation_style" in preferences:
        prompt_parts.append(f"EXPLANATION STYLE: The user prefers: \"{preferences['explanation_style']}\"")
    
    if "confusion_response" in preferences:
        prompt_parts.append(f"WHEN USER IS CONFUSED: {preferences['confusion_response']}")
    
    if "tone_preference" in preferences:
        prompt_parts.append(f"TONE: Use a {preferences['tone_preference'].lower()} tone in your responses.")
    
    if "feedback_style" in preferences:
        prompt_parts.append(f"FEEDBACK APPROACH: {preferences['feedback_style']}")
    
    if "learning_progression" in preferences:
        prompt_parts.append(f"AFTER USER UNDERSTANDS: {preferences['learning_progression']}")
    
    if "interaction_feel" in preferences:
        prompt_parts.append(f"INTERACTION STYLE: Keep interactions {preferences['interaction_feel'].lower()}")
    
    if "promise" in preferences:
        prompt_parts.append(f"PRIORITY: Ensure that {preferences['promise'].lower()}")
    
    if "focus_helpers" in preferences:
        prompt_parts.append(f"USER FOCUSES BEST WITH: {preferences['focus_helpers']}")
    
    if "focus_breakers" in preferences:
        prompt_parts.append(f"AVOID (breaks user focus): {preferences['focus_breakers']}")
    
    if "mistake_response" in preferences:
        prompt_parts.append(f"WHEN USER MAKES MISTAKES: {preferences['mistake_response']}")
    
    if "motivation" in preferences:
        prompt_parts.append(f"USER IS MOTIVATED BY: {preferences['motivation']}")
    
    if "frustration_response" in preferences:
        prompt_parts.append(f"WHEN USER IS FRUSTRATED: {preferences['frustration_response']}")
    
    if "trust_builders" in preferences:
        prompt_parts.append(f"BUILDS TRUST: {preferences['trust_builders']}")
    
    if "kindness_boundaries" in preferences:
        prompt_parts.append(f"AVOID DOING (feels off to this user): {preferences['kindness_boundaries']}")
    
    prompt_parts.extend([
        "",
        "CORE INSTRUCTIONS:",
        "1. Answer based on the provided context",
        "2. Adapt your communication style to match the user's stated preferences above",
        "3. If information is not available, say so clearly",
        "4. Remember and apply the user's preferences consistently throughout the conversation"
    ])
    
    return "\n".join(prompt_parts)


# ========================================
# RAG CHATBOT CLASS
# ========================================
class RAGChatbot:
    """RAG Chatbot with MongoDB Atlas Vector Search and GPT-5.1"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.embedding_model = "text-embedding-3-large"
        self.chat_model = "gpt-5.1"
        self.embedding_cache = {}
    
    def get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding with caching"""
        cache_key = hashlib.md5(query.encode()).hexdigest()
        
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=query
        )
        embedding = response.data[0].embedding
        self.embedding_cache[cache_key] = embedding
        
        return embedding
    
    def retrieve(self, query: str, top_k: int = 20) -> List[Dict]:
        """Vector search using MongoDB Atlas"""
        query_embedding = self.get_query_embedding(query)
        collection = get_embeddings_collection()
        
        # MongoDB Atlas Vector Search aggregation pipeline
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": top_k * 10,
                    "limit": top_k
                }
            },
            {
                "$project": {
                    "page_number": 1,
                    "original_text": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        
        results = list(collection.aggregate(pipeline))
        
        top_pages = []
        for doc in results:
            top_pages.append({
                'page_number': doc.get('page_number'),
                'similarity': doc.get('score', 0),
                'original_text': doc.get('original_text', ''),
            })
        
        return top_pages
    
    def generate_response(self, query: str, retrieved_pages: List[Dict], 
                         system_prompt: str, max_context_length: int = 15000) -> str:
        """Generate response using GPT-5.1"""
        context = self._build_context(retrieved_pages, max_context_length)
        
        full_input = f"""{system_prompt}

QUESTION: {query}

CONTEXT:
{context}

Provide a helpful answer following the user's preferences."""

        result = self.client.responses.create(
            model=self.chat_model,
            input=full_input,
            reasoning={"effort": "medium"},
            text={"verbosity": "medium"}
        )
        
        return result.output_text
    
    def _build_context(self, retrieved_pages: List[Dict], max_length: int) -> str:
        """Build context with smart truncation"""
        context_parts = []
        current_length = 0
        
        for page_data in retrieved_pages:
            page_num = page_data.get('page_number')
            original_text = page_data.get('original_text', '')
            
            header = f"\n--- Page {page_num} ---\n"
            estimated_tokens = len(header + original_text) // 4
            
            if current_length + estimated_tokens > max_length:
                remaining_chars = (max_length - current_length) * 4
                if remaining_chars > 500:
                    truncated_text = original_text[:remaining_chars] + "\n[...]"
                    context_parts.append(header + truncated_text)
                break
            
            context_parts.append(header + original_text)
            current_length += estimated_tokens
        
        return '\n'.join(context_parts)


# ========================================
# LOGIN PAGE
# ========================================
def show_login_page():
    """Display the login/registration page"""
    st.markdown('<h1 class="main-header">RAG Chatbot</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Your personalized document assistant</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        tab1, tab2 = st.tabs(["Login", "New User"])
        
        with tab1:
            st.markdown("### Welcome Back")
            st.write("Enter your User ID to continue:")
            
            user_id_input = st.text_input(
                "User ID",
                placeholder="e.g., A1B2C3D4",
                key="login_id",
                label_visibility="collapsed"
            )
            
            if st.button("Login", type="primary", use_container_width=True):
                if user_id_input:
                    user_id_clean = user_id_input.strip().upper()
                    if user_exists(user_id_clean):
                        st.session_state.user_id = user_id_clean
                        user_data = load_user_data(user_id_clean)
                        st.session_state.chat_history = user_data.get('chat_history', [])
                        st.session_state.preferences = user_data.get('preferences', {})
                        st.session_state.survey_completed = user_data.get('survey_completed', False)
                        st.session_state.logged_in = True
                        st.rerun()
                    else:
                        st.error("User ID not found. Please check your ID or create a new account.")
                else:
                    st.warning("Please enter your User ID")
        
        with tab2:
            st.markdown("### Create New Account")
            st.write("Click the button below to generate your unique User ID:")
            
            if st.button("Generate My User ID", type="primary", use_container_width=True):
                new_id = generate_user_id()
                st.session_state.generated_id = new_id
            
            if 'generated_id' in st.session_state:
                st.markdown(f"""
                    <div class="user-id-display">
                        Your User ID: <strong>{st.session_state.generated_id}</strong>
                    </div>
                """, unsafe_allow_html=True)
                
                st.warning("**Save this ID!** You'll need it to access your chat history in the future.")
                
                if st.button("I've saved my ID - Continue to Setup", use_container_width=True):
                    st.session_state.user_id = st.session_state.generated_id
                    st.session_state.chat_history = []
                    st.session_state.preferences = {}
                    st.session_state.survey_completed = False
                    st.session_state.logged_in = True
                    st.session_state.survey_step = 0
                    # Save initial user to MongoDB
                    save_user_data(st.session_state.user_id, {
                        'chat_history': [],
                        'preferences': {},
                        'survey_completed': False,
                        'created_at': datetime.now().isoformat()
                    })
                    del st.session_state.generated_id
                    st.rerun()


# ========================================
# SURVEY PAGE
# ========================================
def show_survey_page():
    """Display the onboarding survey"""
    st.markdown('<h1 class="main-header">Personalize Your Experience</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Help me understand how you learn best</p>', unsafe_allow_html=True)
    
    if 'survey_responses' not in st.session_state:
        st.session_state.survey_responses = {}
    if 'survey_step' not in st.session_state:
        st.session_state.survey_step = 0
    
    questions = list(SURVEY_QUESTIONS.keys())
    total_questions = len(questions)
    current_step = st.session_state.survey_step
    
    progress = (current_step / total_questions) * 100
    st.markdown(f"""
        <div class="progress-bar">
            <div class="progress-fill" style="width: {progress}%"></div>
        </div>
        <p style="text-align: center; color: #666;">Question {current_step + 1} of {total_questions}</p>
    """, unsafe_allow_html=True)
    
    if current_step < total_questions:
        question_key = questions[current_step]
        question_data = SURVEY_QUESTIONS[question_key]
        
        st.markdown(f'<div class="survey-question">', unsafe_allow_html=True)
        st.markdown(f"### {question_data['question']}")
        
        previous_answer = st.session_state.survey_responses.get(question_key, None)
        default_index = 0
        if previous_answer and previous_answer in question_data['options']:
            default_index = question_data['options'].index(previous_answer)
        
        answer = st.radio(
            "Select one:",
            question_data['options'],
            index=default_index,
            key=f"survey_{question_key}",
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if current_step > 0:
                if st.button("Previous", use_container_width=True):
                    st.session_state.survey_responses[question_key] = answer
                    st.session_state.survey_step -= 1
                    st.rerun()
        
        with col3:
            if current_step < total_questions - 1:
                if st.button("Next", type="primary", use_container_width=True):
                    st.session_state.survey_responses[question_key] = answer
                    st.session_state.survey_step += 1
                    st.rerun()
            else:
                if st.button("Finish Setup", type="primary", use_container_width=True):
                    st.session_state.survey_responses[question_key] = answer
                    st.session_state.preferences = st.session_state.survey_responses.copy()
                    st.session_state.survey_completed = True
                    # Save to MongoDB
                    save_user_data(st.session_state.user_id, {
                        'chat_history': st.session_state.chat_history,
                        'preferences': st.session_state.preferences,
                        'survey_completed': True
                    })
                    st.rerun()
    
    st.divider()
    st.markdown(f"**Your User ID:** `{st.session_state.user_id}`")
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.user_id = None
        st.rerun()


# ========================================
# CHAT PAGE
# ========================================
def show_chat_page():
    """Display the main chat interface"""
    
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        st.markdown(f"**ID:** `{st.session_state.user_id}`")
    with col2:
        st.markdown('<h1 class="main-header" style="font-size: 2rem;">RAG Chatbot</h1>', unsafe_allow_html=True)
    with col3:
        if st.button("Logout"):
            # Save to MongoDB before logout
            save_user_data(st.session_state.user_id, {
                'chat_history': st.session_state.chat_history,
                'preferences': st.session_state.preferences,
                'survey_completed': st.session_state.survey_completed
            })
            st.session_state.logged_in = False
            st.session_state.user_id = None
            st.session_state.chat_history = []
            st.rerun()
    
    st.divider()
    
    system_prompt = build_personalized_prompt(st.session_state.preferences)
    chatbot = RAGChatbot(OPENAI_API_KEY)
    
    chat_container = st.container()
    with chat_container:
        for chat in st.session_state.chat_history:
            st.markdown(f'<div class="user-message">{chat["query"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="assistant-message">{chat["answer"]}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    query = st.text_input(
        "Your question:",
        placeholder="Type your question here...",
        key="query_input",
        label_visibility="collapsed"
    )
    
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        submit_clicked = st.button("Ask", type="primary", use_container_width=True)
    
    if submit_clicked and query:
        with st.spinner("Thinking..."):
            retrieved_pages = chatbot.retrieve(query, TOP_K_PAGES)
            answer = chatbot.generate_response(query, retrieved_pages, system_prompt, MAX_CONTEXT_LENGTH)
        
        # Add to history
        st.session_state.chat_history.append({
            'query': query,
            'answer': answer,
            'timestamp': datetime.now().isoformat()
        })
        
        # Save to MongoDB
        save_user_data(st.session_state.user_id, {
            'chat_history': st.session_state.chat_history,
            'preferences': st.session_state.preferences,
            'survey_completed': st.session_state.survey_completed
        })
        
        st.rerun()
    
    if st.session_state.chat_history:
        st.markdown("---")
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                # Save cleared history to MongoDB
                save_user_data(st.session_state.user_id, {
                    'chat_history': [],
                    'preferences': st.session_state.preferences,
                    'survey_completed': st.session_state.survey_completed
                })
                st.rerun()


# ========================================
# MAIN APP
# ========================================
def main():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'preferences' not in st.session_state:
        st.session_state.preferences = {}
    if 'survey_completed' not in st.session_state:
        st.session_state.survey_completed = False
    
    if OPENAI_API_KEY == "your-openai-api-key-here":
        st.error("Please set your OpenAI API key in the code (line 18)")
        st.stop()
    
    if MONGODB_URI == "mongodb+srv://username:password@cluster.xxxxx.mongodb.net/?retryWrites=true&w=majority":
        st.error("Please set your MongoDB URI in the code (line 19)")
        st.stop()
    
    if not st.session_state.logged_in:
        show_login_page()
    elif not st.session_state.survey_completed:
        show_survey_page()
    else:
        show_chat_page()


if __name__ == "__main__":
    main()