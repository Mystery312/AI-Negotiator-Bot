import json
import re
import logging
import csv
import uuid
from typing import Dict, List, Any
from openai import OpenAI
import pdfplumber
import os
from dotenv import load_dotenv
from app.graph import upsert_turn

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def label_text(text: str) -> Dict[str, str]:
    """
    Classify negotiation speech using GPT-4o-mini.
    
    Args:
        text: The input text to classify
        
    Returns:
        Dict with "move_type", "move", and "pd" keys
    """
    try:
        prompt = f"""Classify this speech. Return JSON {{"move_type", "pd"}} where 
move_type ∈ [concession, threat, info_share, cooperate, defect]
and pd is C if move_type ∈ [concession, info_share, cooperate]
else D.

Text: "{text}"

Return only the JSON object:"""

        logger.info(f"Sending request to OpenAI for text: '{text[:50]}...'")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=100
        )
        
        # Extract JSON from response
        result_text = response.choices[0].message.content.strip()
        logger.info(f"OpenAI response: {result_text}")
        
        # Try to parse JSON, handle potential formatting issues
        try:
            result = json.loads(result_text)
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract JSON from the response
            json_match = re.search(r'\{.*\}', result_text)
            if json_match:
                result = json.loads(json_match.group())
            else:
                # Fallback to default values
                logger.warning(f"Failed to parse JSON from response: {result_text}")
                result = {"move_type": "info_share", "pd": "C"}
        
        # Validate the response
        valid_move_types = ["concession", "threat", "info_share", "cooperate", "defect"]
        if result.get("move_type") not in valid_move_types:
            logger.warning(f"Invalid move_type: {result.get('move_type')}, using default")
            result["move_type"] = "info_share"
        
        # Set pd based on move_type
        cooperative_moves = ["concession", "info_share", "cooperate"]
        if result.get("move_type") in cooperative_moves:
            result["pd"] = "C"
        else:
            result["pd"] = "D"
        
        # Ensure we have both move_type and move fields
        result["move"] = result["move_type"]
        
        logger.info(f"Labeled text: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error labeling text: {e}")
        # Return fallback values with both move_type and move fields
        fallback = {"move_type": "info_share", "move": "info_share", "pd": "C"}
        logger.info(f"Using fallback values: {fallback}")
        return fallback

def csv_to_turns(csv_path: str, conv_id: str) -> List[Dict[str, str]]:
    """
    Extract conversation turns from a CSV file.
    
    Args:
        csv_path: Path to the CSV file
        conv_id: Conversation identifier
        
    Returns:
        List of dictionaries with speaker, text, and conv_id
    """
    turns = []
    
    try:
        with open(csv_path, newline='', encoding="utf-8") as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                # Validate required columns
                if "speaker" not in row or "text" not in row:
                    logger.warning(f"Skipping row missing required columns: {row}")
                    continue
                
                turn = {
                    "speaker": row["speaker"].strip(),
                    "text": row["text"].strip(),
                    "conv_id": conv_id
                }
                
                # Only add if we have both speaker and text
                if turn["speaker"] and turn["text"]:
                    turns.append(turn)
            
            logger.info(f"Extracted {len(turns)} turns from CSV: {csv_path}")
            return turns
            
    except Exception as e:
        logger.error(f"Error processing CSV {csv_path}: {e}")
        return []

def pdf_to_turns(pdf_path: str, conv_id: str) -> List[Dict[str, str]]:
    """
    Extract conversation turns from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        conv_id: Conversation identifier
        
    Returns:
        List of dictionaries with speaker, text, and conv_id
    """
    turns = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Extract text from all pages
            full_text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n"
            
            # Split by speaker pattern
            # Regex: ([A-Z][A-Z\s]+):\s*
            # This matches patterns like "SPEAKER A:", "CHAIRMAN:", etc.
            speaker_pattern = r'([A-Z][A-Z\s]+):\s*'
            
            # Split the text by speaker pattern
            parts = re.split(speaker_pattern, full_text)
            
            # Process the parts to extract speaker-text pairs
            for i in range(1, len(parts), 2):  # Start from 1 to skip the first empty part
                if i + 1 < len(parts):
                    speaker = parts[i].strip()
                    text = parts[i + 1].strip()
                    
                    # Clean up speaker name (remove extra spaces)
                    speaker = re.sub(r'\s+', ' ', speaker)
                    
                    # Only add if we have both speaker and text
                    if speaker and text:
                        turn = {
                            "speaker": speaker,
                            "text": text,
                            "conv_id": conv_id
                        }
                        turns.append(turn)
            
            logger.info(f"Extracted {len(turns)} turns from PDF: {pdf_path}")
            return turns
            
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {e}")
        return []

def process_file_with_labels(file_path: str, conv_id: str) -> List[Dict[str, Any]]:
    """
    Process file (PDF or CSV) and add labels to each turn.
    
    Args:
        file_path: Path to the file (PDF or CSV)
        conv_id: Conversation identifier
        
    Returns:
        List of dictionaries with speaker, text, conv_id, move_type, and pd
    """
    if file_path.lower().endswith('.csv'):
        turns = csv_to_turns(file_path, conv_id)
    else:
        turns = pdf_to_turns(file_path, conv_id)
    
    labeled_turns = []
    
    for turn in turns:
        # Add labels to the turn
        labels = label_text(turn["text"])
        labeled_turn = {
            **turn,
            "move_type": labels["move_type"],
            "pd": labels["pd"]
        }
        labeled_turns.append(labeled_turn)
    
    logger.info(f"Processed {len(labeled_turns)} labeled turns from file: {file_path}")
    return labeled_turns

def process_pdf_with_labels(pdf_path: str, conv_id: str) -> List[Dict[str, Any]]:
    """
    Process PDF and add labels to each turn.
    
    Args:
        pdf_path: Path to the PDF file
        conv_id: Conversation identifier
        
    Returns:
        List of dictionaries with speaker, text, conv_id, move_type, and pd
    """
    turns = pdf_to_turns(pdf_path, conv_id)
    labeled_turns = []
    
    for turn in turns:
        # Add labels to the turn
        labels = label_text(turn["text"])
        labeled_turn = {
            **turn,
            "move_type": labels["move_type"],
            "pd": labels["pd"]
        }
        labeled_turns.append(labeled_turn)
    
    logger.info(f"Processed {len(labeled_turns)} labeled turns from PDF: {pdf_path}")
    return labeled_turns

def save_turns_to_json(turns: List[Dict[str, Any]], output_path: str):
    """
    Save turns to a JSON file.
    
    Args:
        turns: List of turn dictionaries
        output_path: Path to save the JSON file
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(turns, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(turns)} turns to {output_path}")
    except Exception as e:
        logger.error(f"Error saving turns to {output_path}: {e}")

def load_file_to_graph(file_path: str, conv_id: str):
    """
    Load file (PDF or CSV) directly into the graph database.
    
    Args:
        file_path: Path to the file (PDF or CSV)
        conv_id: Conversation identifier
    """
    try:
        if file_path.lower().endswith('.csv'):
            # Process CSV file
            with open(file_path, newline='', encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for idx, row in enumerate(reader, start=1):
                    tag = label_text(row["text"])
                    upsert_turn(
                        conv_id=conv_id,
                        speaker=row["speaker"],
                        text=row["text"],
                        move=tag["move"],
                        pd=tag["pd"],
                    )
                    if idx % 20 == 0:
                        logger.info(f"Loaded {idx} rows…")
        else:
            # Process PDF file
            turns = pdf_to_turns(file_path, conv_id)
            for turn in turns:
                tag = label_text(turn["text"])
                upsert_turn(
                    conv_id=conv_id,
                    speaker=turn["speaker"],
                    text=turn["text"],
                    move=tag["move"],
                    pd=tag["pd"]
                )
        
        logger.info(f"Successfully loaded file {file_path} into graph database")
        
    except Exception as e:
        logger.error(f"Error loading file {file_path} to graph: {e}")
        raise

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python -m app.ingest <pdf_or_csv_path> <conv_id>")
        sys.exit(1)

    path, conv_id = sys.argv[1], sys.argv[2]

    try:
        # Load file directly into graph database
        load_file_to_graph(path, conv_id)
        print(f"Successfully processed {path} and loaded into graph database")
        
    except Exception as e:
        print(f"Error processing {path}: {e}")
        sys.exit(1)
