# src/utils/formatter.py


def format_answer(answer_text: str, image_ids: list[str]) -> str:
    """
    Format answer according to submission requirements.

    Args:
        answer_text: Answer text with <PIC> placeholders
        image_ids: List of image IDs corresponding to <PIC> placeholders

    Returns:
        Formatted answer string:
        - No images: plain text
        - With images: '"text with <PIC>", ["id1", "id2"]'
    """
    # Strip all newlines to match standard answer format
    answer_text = answer_text.replace('\n', ' ').replace('\r', ' ')
    # Collapse multiple spaces
    answer_text = ' '.join(answer_text.split())

    if not image_ids:
        return answer_text

    # Format with images: '"text", ["id1", "id2"]'
    image_list = str(image_ids).replace("'", '"')
    return f'"{answer_text}", {image_list}'
