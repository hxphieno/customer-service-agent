"""
Chunker module: splits manual text into parent and child chunks.

Parent chunks: split by # headings
Child chunks: split parent chunks by max_chars, inheriting parent metadata
"""

import re
from typing import Dict, List, Tuple


def split_into_parent_chunks(
    text: str, product: str, pic_position_map: Dict[int, str]
) -> List[Dict]:
    """
    Split text by # headings into parent chunks.

    Args:
        text: Manual text content
        product: Product name
        pic_position_map: {pic_index: image_id}

    Returns:
        List of parent chunk dicts with keys:
        - product, chapter, parent_id, text, image_ids, chunk_type
    """
    # Split by # headings
    sections = re.split(r'\n(?=#\s)', text)
    chunks = []

    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Extract chapter name
        match = re.match(r'^#\s+(.+?)(?:\n|$)', section)
        if match:
            chapter = match.group(1).strip()
            section_text = section[match.end():].strip()
        else:
            chapter = "前言"
            section_text = section

        # Find image IDs in this section
        pic_count = section_text.count('<PIC>')
        image_ids = []

        # Map <PIC> occurrences to image IDs
        current_pic_index = 0
        for i in range(len(chunks)):
            current_pic_index += chunks[i]["text"].count('<PIC>')

        for i in range(pic_count):
            pic_idx = current_pic_index + i
            if pic_idx in pic_position_map:
                image_ids.append(pic_position_map[pic_idx])

        parent_id = f"{product}_{chapter}"

        chunks.append({
            "product": product,
            "chapter": chapter,
            "parent_id": parent_id,
            "text": section_text,
            "image_ids": image_ids,
            "chunk_type": "parent"
        })

    return chunks


def split_parent_into_children(
    parent: Dict, max_chars: int = 500
) -> List[Dict]:
    """
    Split a parent chunk into child chunks by max_chars.

    Args:
        parent: Parent chunk dict
        max_chars: Maximum characters per child chunk

    Returns:
        List of child chunk dicts, inheriting parent metadata
    """
    text = parent["text"]
    children = []

    # Split by sentences to avoid breaking mid-sentence
    sentences = re.split(r'([。！？\n])', text)

    current_chunk = ""
    chunk_index = 0

    for i in range(0, len(sentences), 2):
        sentence = sentences[i]
        delimiter = sentences[i + 1] if i + 1 < len(sentences) else ""
        full_sentence = sentence + delimiter

        if len(current_chunk) + len(full_sentence) > max_chars and current_chunk:
            # Save current chunk
            child_id = f"{parent['parent_id']}_child_{chunk_index}"
            children.append({
                "product": parent["product"],
                "chapter": parent["chapter"],
                "parent_id": parent["parent_id"],
                "child_id": child_id,
                "text": current_chunk.strip(),
                "image_ids": parent["image_ids"],
                "chunk_type": "child"
            })
            chunk_index += 1
            current_chunk = full_sentence
        else:
            current_chunk += full_sentence

    # Add remaining text
    if current_chunk.strip():
        child_id = f"{parent['parent_id']}_child_{chunk_index}"
        children.append({
            "product": parent["product"],
            "chapter": parent["chapter"],
            "parent_id": parent["parent_id"],
            "child_id": child_id,
            "text": current_chunk.strip(),
            "image_ids": parent["image_ids"],
            "chunk_type": "child"
        })

    return children


def chunk_manual(parsed_manual: Dict) -> Tuple[List[Dict], List[Dict]]:
    """
    Complete chunking pipeline: parent chunks + child chunks.

    Args:
        parsed_manual: Output from parser.parse_manual()

    Returns:
        (parent_chunks, child_chunks)
    """
    product = parsed_manual["product"]
    text = parsed_manual["text"]
    pic_position_map = parsed_manual["pic_position_map"]

    # Step 1: Create parent chunks
    parent_chunks = split_into_parent_chunks(text, product, pic_position_map)

    # Step 2: Create child chunks from each parent
    child_chunks = []
    for parent in parent_chunks:
        children = split_parent_into_children(parent, max_chars=500)
        child_chunks.extend(children)

    return parent_chunks, child_chunks
