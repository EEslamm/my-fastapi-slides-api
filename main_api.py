import os
import re
import asyncio
import hashlib
import tempfile
from io import BytesIO
from typing import List, Optional, Dict, Any
import pdfplumber
from dotenv import load_dotenv
import google.generativeai as genai
import edge_tts
from yt_dlp import YoutubeDL
from functools import lru_cache
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

# --- Configuration ---
# تأكد أنك بتستخدم مفتاح API حقيقي ومفعل
os.environ["GOOGLE_API_KEY"] = "AIzaSyAYUAYJw6ca4HltF_h_kOFjvLaaaf9SUEA"
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

try:
    model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')
except Exception as e:
    raise RuntimeError(f"Failed to initialize GenerativeModel. Check API key and model name. Error: {e}")

# --- Prompts (Copied from your script) ---
explain_prompt = """
### **Transform Educational Content into Interactive Slides**
Create slides appropriate for {level} learners. Follow these guidelines:

{level_instructions}

Format each slide with:
- Title starting with "### Slide X: [Title]"
- Content matching the expertise level
- Include a topic for video recommendations in a separate line in the format "VIDEO_TOPIC: topic keywords" (This line will be removed from the displayed content)

**Example Format:**
### Slide 1: [Title]
[Level-appropriate content]

VIDEO_TOPIC: [topic]

---

**Input Text:**
"""

LEVEL_INSTRUCTIONS = {
    "Basic": """
    BASIC LEVEL REQUIREMENTS:
    - Explain concepts like teaching to complete beginners
    - Use simple language and short sentences
    - Add multiple examples for each concept
    - Include definitions for technical terms
    - Break complex ideas into step-by-step explanations
    - Add "Key Point" boxes for important concepts
    """,
    "Intermediate": """
    INTERMEDIATE LEVEL REQUIREMENTS:
    - Balance depth and accessibility
    - Assume basic domain knowledge
    - Use technical terms with brief explanations
    - Include 1-2 examples per complex concept
    - Highlight connections between concepts
    """
}

quiz_prompt = """
Generate mixed question types (MCQs and True/False) based on slide count. Follow these rules:
- Create 1-3 questions per slide
- Mix question types naturally
- Follow formats:

MCQ Format:
1. [Question]
    A. [Option]
    B. [Option]
    C. [Option]
    D. [Option]
    Correct Answer: [Letter]. Explanation: [Context from slides]

True/False Format:
2. [Statement]
    A. True
    B. False
    Correct Answer: [A/B]. Explanation: [Context from slides]

Include explanations referencing specific slides. Ensure unambiguous answers.

**Input Text:**
"""


# --- Utility Functions (Refactored for API) ---
def clean_text(text: str) -> str:
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    return '\n'.join(lines)


def extract_text_from_pdf(pdf_file_bytes: bytes) -> str:
    text = ""
    try:
        with pdfplumber.open(BytesIO(pdf_file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n--- PAGE BREAK ---\n\n"
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")  # Log error
        raise HTTPException(status_code=500, detail=f"Error extracting text from PDF: {str(e)}")


def split_text(text: str, min_chunk_size: int = 2000, max_chunk_size: int = 4000) -> List[str]:
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if (len(current_chunk) + len(para) > max_chunk_size and
                    len(current_chunk) >= min_chunk_size):
            chunks.append(current_chunk.strip())
            current_chunk = para
        else:
            current_chunk += ('\n\n' if current_chunk else '') + para  # Avoid leading newline for first para
    if current_chunk.strip():  # Ensure non-empty chunk
        chunks.append(current_chunk.strip())
    return chunks


def prepare_text_for_tts(text: str) -> str:
    text = re.sub(r'[#*`_~\-–—]', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\bfig\.\s', 'figure ', text, flags=re.IGNORECASE)
    text = re.sub(r'\be\.g\.\s', 'for example, ', text, flags=re.IGNORECASE)
    text = re.sub(r'\bi\.e\.\s', 'that is, ', text, flags=re.IGNORECASE)
    text = re.sub(r'\s\([^)]{1,4}\)', '', text)
    text = re.sub(r'\s\[[^\]]{1,4}\]', '', text)
    text = re.sub(r'\(([^)]{5,})\)', r'\1', text)  # Keep longer parenthetical content
    text = re.sub(r'\[([^\]]{5,})\]', r'\1', text)  # Keep longer bracketed content
    return text.strip()


# --- AI Service Function (Refactored) ---
def run_gemini_task_api(prompt_template: str, text_input: str) -> str:
    full_prompt = f"{prompt_template}\n\n{text_input}"
    try:
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        print(f"Error processing text with Gemini: {e}")
        raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")


# --- In-memory Caches (Simple global dicts) ---
# For production, consider Redis or a more robust caching solution.
api_tts_cache: Dict[str, bytes] = {}
api_video_cache: Dict[str, List[Dict[str, Any]]] = {}


# --- TTS Module (Refactored) ---
def get_tts_cache_key(text: str, voice: str, rate: str) -> str:
    key_content = f"{text}_{voice}_{rate}"
    return hashlib.md5(key_content.encode()).hexdigest()


async def text_to_speech_async_api(text: str, voice: str = "en-US-ChristopherNeural", rate: str = "+0%") -> bytes:
    cleaned_text = prepare_text_for_tts(text)
    if not cleaned_text:  # Handle empty text after cleaning
        raise ValueError("Cannot generate speech for empty text.")

    cache_key = get_tts_cache_key(cleaned_text, voice, rate)

    if cache_key in api_tts_cache:
        return api_tts_cache[cache_key]

    try:
        communicate = edge_tts.Communicate(cleaned_text, voice, rate=rate)
        # Use a temporary file for edge_tts to write to, then read and delete.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file_obj:
            temp_path = temp_audio_file_obj.name

        await communicate.save(temp_path)  # edge_tts saves to the file path

        with open(temp_path, 'rb') as f:
            audio_data = f.read()

        os.unlink(temp_path)  # Clean up the temporary file
        api_tts_cache[cache_key] = audio_data
        return audio_data
    except Exception as e:
        if 'temp_path' in locals() and os.path.exists(temp_path):  # Ensure temp file is cleaned up on error
            os.unlink(temp_path)
        print(f"Error generating audio with Edge TTS: {e}")
        raise HTTPException(status_code=500, detail=f"TTS generation error: {str(e)}")


def chunk_text_for_tts(text: str, max_chars: int = 1000) -> List[str]:
    cleaned_text = prepare_text_for_tts(text)  # prepare_text_for_tts first
    if not cleaned_text:
        return []
    if len(cleaned_text) <= max_chars:
        return [cleaned_text]

    sentences = re.split(r'(?<=[.!?])\s+', cleaned_text)  # Split by sentences
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_chars:  # +1 for space
            current_chunk += (" " if current_chunk else "") + sentence
        else:
            if current_chunk:  # Add previous chunk
                chunks.append(current_chunk.strip())
            current_chunk = sentence  # Start new chunk
            # Handle very long sentences that exceed max_chars
            while len(current_chunk) > max_chars:
                chunks.append(current_chunk[:max_chars])
                current_chunk = current_chunk[max_chars:]

    if current_chunk.strip():  # Add the last chunk
        chunks.append(current_chunk.strip())
    return chunks


# --- Video Module (Refactored to return YouTube URLs directly) ---
# --- (محتويات الكاش api_video_cache وتعاريف الـ Pydantic Model زي SlideDataItem بتكون موجودة في مكانها خارج الدالة) ---

def get_related_videos_api(topic: str, slide_content: Optional[str] = None, max_results: int = 3) -> List[Dict[str, str]]:
    cache_key_base = topic
    if slide_content:
        slide_hash = hashlib.md5(slide_content.encode()).hexdigest()[:8]
        cache_key = f"{cache_key_base}_{slide_hash}"
    else:
        cache_key = cache_key_base

    if cache_key in api_video_cache:
        print(f"DEBUG: Video cache hit for topic: {topic}")
        return api_video_cache[cache_key]

    key_concepts = []
    if slide_content:
        # استخراج النقاط الأساسية من محتوى السلايد
        bullet_points = re.findall(r'-\s*(.*?)(?:\n|$)', slide_content)
        key_concepts = [point.strip() for point in bullet_points if len(point.strip()) > 3][:2]

    # بناء استعلام البحث
    search_terms = [topic] + key_concepts
    final_search_query = " ".join(term for term in search_terms if term) + " educational video"
    # لو الاستعلام طويل جداً، قلّله
    if len(final_search_query) > 100:
        final_search_query = topic + " educational video"

    print(f"DEBUG: Searching YouTube for query: '{final_search_query}'")

    ydl_opts = {
        'quiet': True,              
        'verbose': False,           
        'extract_flat': True,       
        'ignoreerrors': True,       
    }

    videos_found_with_score = [] 
    try:
        with YoutubeDL(ydl_opts) as ydl:
            search_string = f"ytsearch{max_results * 5}:{final_search_query}" 
            search_result = ydl.extract_info(search_string, download=False)
            
            print(f"DEBUG: Full search_result keys: {search_result.keys() if search_result else 'None'}")
            print(f"DEBUG: YouTubeDL search result (first 2 entries): {search_result.get('entries', [])[:2]}")

            if search_result and 'entries' in search_result:
                for entry in search_result.get('entries', []):
                    if isinstance(entry, dict):
                        video_id = str(entry.get('id', ''))
                        video_title = str(entry.get('title', ''))

                        if not video_id or not video_title:
                            continue

                        current_video_url = f"https://www.youtube.com/watch?v={video_id}"
                        
                        if len(video_id) == 11 and \
                           "shorts" not in video_title.lower() and \
                           "playlist" not in video_title.lower():
                            
                            relevance = sum(1 for term in search_terms if term.lower() in video_title.lower())
                            videos_found_with_score.append({
                                'url': current_video_url,
                                'title': video_title,
                                'relevance_score': relevance
                            })
        
        print(f"DEBUG: Total videos found (before max_results filter): {len(videos_found_with_score)}")

        if videos_found_with_score:
            videos_found_with_score.sort(key=lambda x: x['relevance_score'], reverse=True)
            
        # هنا التغيير المهم: نرجع objects بدلاً من strings فقط
        result_videos = []
        for video in videos_found_with_score[:max_results]:
            result_videos.append({
                'url': video['url'],
                'title': video['title']
            })
        
        api_video_cache[cache_key] = result_videos
        return result_videos
        
    except Exception as e:
        print(f"CRITICAL ERROR: Error finding related videos for '{topic}': {e}")
        return []

# --- Quiz Module (Refactored) ---
def parse_quiz_response_api(response_text: str) -> List[Dict[str, Any]]:
    question_blocks = response_text.strip().split("\n\n")
    questions = []

    for block in question_blocks:
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        if not lines:
            continue

        question_match = re.match(r'^\d+\.\s*(.*)', lines[0])
        if not question_match:
            continue

        question_text = question_match.group(1).strip()

        options = []
        option_lines = [line for line in lines[1:] if re.match(r'^[A-Z]\.\s+', line)]

        correct_answer_line = ""
        explanation_text = ""

        for i, line in enumerate(lines):
            if line.startswith("Correct Answer:"):
                correct_answer_line = line
                explanation_match = re.search(r'Explanation:\s*(.*)', correct_answer_line, re.IGNORECASE)
                if explanation_match:
                    explanation_text = explanation_match.group(1).strip()
                    correct_answer_line = re.sub(r'\.?\s*Explanation:.*', '', correct_answer_line,
                                                 flags=re.IGNORECASE).strip()
                break

        if not question_text or not option_lines or not correct_answer_line:
            continue

        correct_answer_match = re.search(r'Correct Answer:\s*([A-Z])', correct_answer_line, re.IGNORECASE)
        if not correct_answer_match:
            continue
        correct_answer_val = correct_answer_match.group(1).upper()

        formatted_options = [opt for opt in option_lines]

        if (len(formatted_options) == 2 or len(formatted_options) == 4):
            questions.append({
                "question": question_text,
                "options": formatted_options,
                "correct": correct_answer_val,
                "explanation": explanation_text
            })
        else:
            print(
                f"Skipping question due to invalid number of options: {question_text} (Found {len(formatted_options)} options)")

    return questions


# --- Pydantic Models for API Request/Response ---
class SlideGenerationRequest(BaseModel):
    level: str = "Intermediate"


class RegenerateSlidesRequest(BaseModel):
    original_lesson_text: str


class TTSRequest(BaseModel):
    text: str
    voice: str = "en-US-ChristopherNeural"
    rate: str = "+0%"


class RelatedVideosRequest(BaseModel):
    topic: str
    slide_content: Optional[str] = None


class AskQuestionRequest(BaseModel):
    slide_content: str
    question: str
    allow_out_of_scope: bool = False
    level: str = "Intermediate"


class QuizGenerationRequest(BaseModel):
    slides_content: List[str]


# Response Models
# **التعديل هنا: SlideDataItem هيشمل youtube_links بدلاً من video_topic**
class SlideDataItem(BaseModel):
    title: str
    content: str
    youtube_links: List[str] = [] # قائمة روابط يوتيوب


class SlideGenerationResponse(BaseModel):
    slides_data: List[SlideDataItem]
    original_lesson_text: str


class QuizQuestionItem(BaseModel):
    question: str
    options: List[str]
    correct: str
    explanation: str


class QuizResponse(BaseModel):
    questions: List[QuizQuestionItem]


class AnswerResponse(BaseModel):
    answer: str


class VideoItem(BaseModel):
    url: str
    title: str


class RelatedVideosResponse(BaseModel):
    videos: List[VideoItem]


# --- FastAPI Application Instance ---
app = FastAPI(title="AI Tutor API")


# --- Helper Function for Slide Generation Logic ---
async def _process_slide_generation_logic(lesson_text: str, level: str, is_regeneration: bool = False) -> Dict[str, Any]:
    level_instructions = LEVEL_INSTRUCTIONS.get(level)
    if not level_instructions:
        raise HTTPException(status_code=400,
                            detail=f"Invalid level: {level}. Choose from {list(LEVEL_INSTRUCTIONS.keys())}.")

    current_explain_prompt_template = explain_prompt.format(
        level=level,
        level_instructions=level_instructions
    )
    if is_regeneration:
        current_explain_prompt_template += "\n\nIMPORTANT: Make explanations MUCH SIMPLER than before. Use extremely basic vocabulary, short sentences, and many examples. Avoid technical terms where possible, and when necessary, define them immediately."

    PRACTICAL_CHUNK_LIMIT = 15000

    presentation_parts = []
    if len(lesson_text) > PRACTICAL_CHUNK_LIMIT:
        chunks = split_text(lesson_text, max_chunk_size=PRACTICAL_CHUNK_LIMIT)
        print(f"Lesson text (length {len(lesson_text)}) split into {len(chunks)} chunks for slide generation.")
        for i, chunk in enumerate(chunks):
            try:
                response_text = run_gemini_task_api(current_explain_prompt_template, chunk)
                if response_text:
                    presentation_parts.append(response_text)
                else:
                    presentation_parts.append(
                        f"### Slide X: Error Processing Chunk\nContent generation for chunk {i + 1} returned empty.\nVIDEO_TOPIC: error")
            except HTTPException as e:
                print(f"Error processing chunk {i + 1} for slides: {e.detail}")
                presentation_parts.append(
                    f"### Slide X: Error Processing Chunk\nContent for this chunk could not be generated: {e.detail}\nVIDEO_TOPIC: error")
    else:
        try:
            response_text = run_gemini_task_api(current_explain_prompt_template, lesson_text)
            if response_text:
                presentation_parts.append(response_text)
        except HTTPException as e:
            raise HTTPException(status_code=e.status_code,
                                detail=f"Failed to generate slides from single block: {e.detail}")

    if not presentation_parts or all("Error Processing Chunk" in part for part in presentation_parts):
        raise HTTPException(status_code=500, detail="No valid slide content generated from Gemini.")

    full_presentation_text = "\n---\n".join(presentation_parts)

    raw_slides_from_gemini = [s.strip() for s in full_presentation_text.split("---") if s.strip()]

    processed_slides_data: List[SlideDataItem] = []
    slide_counter = 1

    for raw_slide_text in raw_slides_from_gemini:
        title_match = re.search(r'### Slide \d+:\s*(.+)', raw_slide_text, re.IGNORECASE)
        title = title_match.group(1).strip() if title_match else f"Slide {slide_counter}"

        topic_match = re.search(r'VIDEO_TOPIC:\s*(.+?)$', raw_slide_text, re.MULTILINE | re.IGNORECASE)
        video_topic = topic_match.group(1).strip() if topic_match else title

        content_cleaned = re.sub(r'VIDEO_TOPIC:\s*.+?$', '', raw_slide_text, flags=re.MULTILINE | re.IGNORECASE)
        content_cleaned = re.sub(r'^### Slide \d+:\s*.*\n?', '', content_cleaned, flags=re.IGNORECASE, count=1).strip()

        if "Error Processing Chunk" in raw_slide_text and "Content for this chunk could not be generated" in raw_slide_text:
            title = f"Slide {slide_counter}: Content Generation Error"

        # **هنا التعديل الأهم في الـ Backend**: بنبحث عن الفيديوهات لكل شريحة ونضيفها
        youtube_videos_info = get_related_videos_api(video_topic, content_cleaned)
        youtube_links = [video['url'] for video in youtube_videos_info] # استخلاص الروابط فقط

        processed_slides_data.append(SlideDataItem(
            title=title,
            content=content_cleaned,
            youtube_links=youtube_links # بنمرر قائمة الروابط هنا
        ))
        slide_counter += 1

    if not processed_slides_data:
        raise HTTPException(status_code=500, detail="Failed to parse any valid slides from the generated content.")

    return {"slides_data": processed_slides_data, "original_lesson_text": lesson_text}


# --- API Endpoints ---

@app.post("/generate_slides/", response_model=SlideGenerationResponse)
async def generate_slides_endpoint(
        level: str = Form("Intermediate"),
        pdf_file: UploadFile = File(...)
):
    """
    Uploads a PDF, extracts text, and generates educational slides based on the specified level.
    """
    if pdf_file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF is supported.")
    try:
        pdf_bytes = await pdf_file.read()
        if not pdf_bytes:
            raise HTTPException(status_code=400, detail="Uploaded PDF file is empty.")

        lesson_text = extract_text_from_pdf(pdf_bytes)
        lesson_text = clean_text(lesson_text)

        if not lesson_text.strip():
            raise HTTPException(status_code=400,
                                detail="No text could be extracted from the PDF, or PDF is image-based.")

        # استدعاء الدالة المساعدة بشكل صحيح
        return await _process_slide_generation_logic(lesson_text, level, is_regeneration=False)

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Unexpected error in /generate_slides/: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during slide generation: {str(e)}")


@app.post("/regenerate_slides_basic/", response_model=SlideGenerationResponse)
async def regenerate_slides_basic_endpoint(request: RegenerateSlidesRequest):
    """
    Regenerates slides from previously extracted text, forcing 'Basic' level for simpler explanations.
    """
    try:
        if not request.original_lesson_text or not request.original_lesson_text.strip():
            raise HTTPException(status_code=400, detail="Original lesson text must be provided and cannot be empty.")

        return await _process_slide_generation_logic(request.original_lesson_text, "Basic", is_regeneration=True)
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Unexpected error in /regenerate_slides_basic/: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during slide regeneration: {str(e)}")


@app.post("/tts/")
async def tts_endpoint(request: TTSRequest):
    """
    Converts provided text to speech.
    Returns audio data as a stream (likely MP3).
    The client should handle playing audio chunks if the text is long.
    This endpoint currently returns audio for the *first* chunk if chunking occurs.
    """
    try:
        content_text = re.sub(r'^### Slide \d+: .*?\n', '', request.text.strip(), count=1)

        if not content_text:
            raise HTTPException(status_code=400, detail="Text for TTS cannot be empty after cleaning.")

        text_chunks = chunk_text_for_tts(content_text)

        if not text_chunks:
            raise HTTPException(status_code=400, detail="No processable text chunks found for TTS.")

        first_chunk_audio_data = await text_to_speech_async_api(
            text_chunks[0],
            voice=request.voice,
            rate=request.rate
        )

        return StreamingResponse(BytesIO(first_chunk_audio_data), media_type="audio/mpeg")

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Unexpected error in /tts/: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during TTS processing: {str(e)}")


@app.post("/related_videos/", response_model=RelatedVideosResponse)
async def related_videos_endpoint(request: RelatedVideosRequest):
    """
    Finds YouTube videos related to a given topic, optionally using slide content for context.
    """
    try:
        if not request.topic or not request.topic.strip():
            raise HTTPException(status_code=400, detail="Search topic must be provided.")
        videos = get_related_videos_api(request.topic, request.slide_content)
        return RelatedVideosResponse(videos=videos)
    except Exception as e:
        print(f"Unexpected error in /related_videos/: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while fetching videos: {str(e)}")


# --- AI Service for Q&A (Refactored) ---
def answer_student_question_api(slide_content: str, question_text: str, allow_out_of_scope: bool, level: str) -> str:
    if allow_out_of_scope:
        qa_full_prompt = f"""
You are an expert educational tutor helping a student understand complex topics.
Answer the following question thoroughly but concisely, even if it's outside the scope of the current slide content.

Your response should:
1. Directly address the student's question with accurate information
2. Use clear, simple language appropriate for educational purposes. Target a {level} level of understanding.
3. Include relevant examples when helpful for understanding.
4. Connect the answer to broader concepts where appropriate.
5. Highlight key terms or concepts using bold formatting.
6. If the question is related to the slide content, prioritize that information.
7. If the question is outside the scope of the slide, provide a helpful answer based on general knowledge.
8. End with a brief check for understanding if the concept is complex.

SLIDE CONTENT (for reference, if question is related):
---
{slide_content}
---

STUDENT'S QUESTION:
{question_text}

Remember to provide a helpful response regardless of whether the question relates directly to the slide content. Match the complexity to the user's selected expertise level of {level}.
"""
    else:
        qa_full_prompt = f"""
You are an expert educational tutor helping a student understand complex topics.
Based on the slide content provided below, answer the student's question thoroughly but concisely.

Your response should:
1. Directly address the student's question with accurate information from the slide.
2. Use clear, simple language appropriate for educational purposes. Target a {level} level of understanding.
3. Include relevant examples when helpful for understanding.
4. Connect the answer to broader concepts where appropriate.
5. Highlight key terms or concepts using bold formatting.
6. Prioritize information from the slide content. If supplemental general knowledge is needed, clearly state it.
7. End with a brief check for understanding if the concept is complex.
8. If the question requires knowledge significantly beyond the slide content, indicate this clearly and explain why.

SLIDE CONTENT:
---
{slide_content}
---

STUDENT'S QUESTION:
{question_text}

Remember to balance depth and clarity in your response. Match the complexity to the user's selected expertise level of {level}.
"""
    try:
        response = model.generate_content(qa_full_prompt)
        return response.text
    except Exception as e:
        print(f"Error in answer_student_question_api with Gemini: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating answer from AI model: {str(e)}")


@app.post("/ask_question/", response_model=AnswerResponse)
async def ask_question_endpoint(request: AskQuestionRequest):
    """
    Answers a student's question based on slide content, with an option to answer out-of-scope questions.
    """
    try:
        if not request.question or not request.question.strip():
            raise HTTPException(status_code=400, detail="Question text cannot be empty.")
        if not request.slide_content and not request.allow_out_of_scope:
            raise HTTPException(status_code=400,
                                detail="Slide content must be provided if not allowing out-of-scope answers.")

        answer = answer_student_question_api(
            request.slide_content,
            request.question,
            request.allow_out_of_scope,
            request.level
        )
        return AnswerResponse(answer=answer)
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Unexpected error in /ask_question/: {e}")
        raise HTTPException(status_code=500,
                            detail=f"An unexpected error occurred while processing the question: {str(e)}")


@app.post("/generate_quiz/", response_model=QuizResponse)
async def generate_quiz_endpoint(request: QuizGenerationRequest):
    """
    Generates a quiz (MCQs and True/False) based on the content of provided slides.
    """
    try:
        if not request.slides_content or not any(s.strip() for s in request.slides_content):
            raise HTTPException(status_code=400,
                                detail="Slide content must be provided and cannot be empty for quiz generation.")

        full_lesson_content = "\n\n--- SLIDE BREAK ---\n\n".join(
            [s for s in request.slides_content if s.strip()]
        )

        if not full_lesson_content.strip():
            raise HTTPException(status_code=400, detail="Combined slide content is empty.")

        current_quiz_prompt_template = quiz_prompt + "\nIMPORTANT: Ensure each question has EXACTLY 2 options (for True/False) or EXACTLY 4 options (for MCQs). Format the 'Correct Answer' line precisely as shown in the examples."

        response_text = run_gemini_task_api(current_quiz_prompt_template, full_lesson_content)

        questions = []
        if response_text:
            questions = parse_quiz_response_api(response_text)

        min_desired_questions = 3
        if len(questions) < min_desired_questions and len(
                request.slides_content) > 0:
            print(f"Initial quiz generation yielded {len(questions)} questions. Retrying for more.")
            retry_quiz_prompt_template = quiz_prompt + f"\nIMPORTANT: Generate at least {min_desired_questions + 2} questions. Ensure each question has EXACTLY 2 options (for True/False) or EXACTLY 4 options (for MCQs). Form"
            # Here you would call run_gemini_task_api again with the retry prompt
            # For simplicity, we'll just return what we have for now.
            # In a real app, you might have a loop or more sophisticated retry logic.

        return QuizResponse(questions=questions)
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Unexpected error in /generate_quiz/: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during quiz generation: {str(e)}")