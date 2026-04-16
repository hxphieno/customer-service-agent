"""
Chunker module: splits manual text into parent and child chunks.

Parent chunks: split by fixed length (1500 chars)
Child chunks: split parent chunks by max_chars, inheriting parent metadata
"""

import re
from typing import Dict, List, Tuple


def split_into_parent_chunks(
    text: str, product: str, pic_position_map: Dict[int, str], max_chars: int = 1500
) -> List[Dict]:
    """
    Split text by fixed length into parent chunks, respecting sentence boundaries.

    Args:
        text: Manual text content
        product: Product name
        pic_position_map: {pic_index: image_id}
        max_chars: Maximum characters per parent chunk

    Returns:
        List of parent chunk dicts with keys:
        - product, chapter, parent_id, text, image_ids, chunk_type
    """
    # Split by sentence boundaries
    sentences = re.split(r'([。！？\n])', text)
    # Recombine sentence + delimiter
    parts = []
    for i in range(0, len(sentences), 2):
        if i + 1 < len(sentences):
            parts.append(sentences[i] + sentences[i + 1])
        else:
            parts.append(sentences[i])

    parent_chunks = []
    current_text = ""
    chunk_index = 0
    global_pic_index = 0

    for part in parts:
        if len(current_text) + len(part) > max_chars and current_text:
            # Calculate image_ids for current chunk
            pic_count = current_text.count("<PIC>")
            image_ids = [
                pic_position_map[global_pic_index + i]
                for i in range(pic_count)
                if (global_pic_index + i) in pic_position_map
            ]
            global_pic_index += pic_count

            parent_id = f"{product}_p{chunk_index}"
            parent_chunks.append({
                "product": product,
                "chapter": f"part_{chunk_index}",
                "parent_id": parent_id,
                "text": current_text.strip(),
                "image_ids": image_ids,
                "chunk_type": "parent",
            })
            current_text = part
            chunk_index += 1
        else:
            current_text += part

    # Handle last chunk
    if current_text.strip():
        pic_count = current_text.count("<PIC>")
        image_ids = [
            pic_position_map[global_pic_index + i]
            for i in range(pic_count)
            if (global_pic_index + i) in pic_position_map
        ]
        parent_id = f"{product}_p{chunk_index}"
        parent_chunks.append({
            "product": product,
            "chapter": f"part_{chunk_index}",
            "parent_id": parent_id,
            "text": current_text.strip(),
            "image_ids": image_ids,
            "chunk_type": "parent",
        })

    return parent_chunks


def split_parent_into_children(
    parent: Dict, max_chars: int = 1000
) -> List[Dict]:
    """
    Split a parent chunk into child chunks by max_chars.

    Args:
        parent: Parent chunk dict
        max_chars: Maximum characters per child chunk

    Returns:
        List of child chunk dicts, each inheriting parent's image_ids
    """
    text = parent["text"]

    # If text is short enough, return as single child
    if len(text) <= max_chars:
        child = parent.copy()
        child["chunk_type"] = "child"
        child["child_id"] = f"{parent['parent_id']}_c0"
        return [child]

    # Split by sentence boundaries
    sentences = re.split(r'([。！？\n])', text)
    parts = []
    for i in range(0, len(sentences), 2):
        if i + 1 < len(sentences):
            parts.append(sentences[i] + sentences[i + 1])
        else:
            parts.append(sentences[i])

    children = []
    current_text = ""
    chunk_index = 0

    for part in parts:
        if len(current_text) + len(part) > max_chars and current_text:
            child = parent.copy()
            child["chunk_type"] = "child"
            child["child_id"] = f"{parent['parent_id']}_c{chunk_index}"
            child["text"] = current_text.strip()
            children.append(child)

            current_text = part
            chunk_index += 1
        else:
            current_text += part

    # Add remaining text
    if current_text.strip():
        child = parent.copy()
        child["chunk_type"] = "child"
        child["child_id"] = f"{parent['parent_id']}_c{chunk_index}"
        child["text"] = current_text.strip()
        children.append(child)

    return children


def chunk_manual(parsed: Dict) -> Tuple[List[Dict], List[Dict]]:
    """
    Execute complete parent-child chunking for a single manual.

    Args:
        parsed: Parsed manual dict from parser.parse_manual()

    Returns:
        Tuple of (parent_chunks, child_chunks)
    """
    parents = split_into_parent_chunks(
        parsed["text"], parsed["product"], parsed["pic_position_map"]
    )
    children = []
    for parent in parents:
        children.extend(split_parent_into_children(parent))
    return parents, children
