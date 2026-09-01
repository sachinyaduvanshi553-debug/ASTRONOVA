import re

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        if self._pageNumber == 1:
            # Suppress header/footer on title page
            return

        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#4A5568"))

        # Header
        self.drawString(54, 11 * 72 - 36, "AstroNova Project Report — ISRO SOLEXS Payload")
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(54, 11 * 72 - 42, 8.5 * 72 - 54, 11 * 72 - 42)

        # Footer
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * 72 - 54, 36, page_str)
        self.drawString(54, 36, "CONFIDENTIAL & PROPRIETARY — ACADEMIC REPORT")
        self.line(54, 48, 8.5 * 72 - 54, 48)

        self.restoreState()


def clean_markdown_inline(text):
    # Convert bold **text** to <b>text</b>
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
    # Convert italic *text* or _text_ to <i>text</i>
    text = re.sub(r"\*(.*?)\*", r"<i>\1</i>", text)
    # Convert inline code `text` to font tag
    text = re.sub(r"`(.*?)`", r'<font face="Courier" color="#C7254E" size="9">\1</font>', text)

    # Handle markdown links [text](url)
    def link_replacer(match):
        label, url = match.group(1), match.group(2)
        if url.startswith("#"):
            # Internal anchor link (e.g. Table of contents)
            return f"<u>{label}</u>"
        elif url.startswith("http://") or url.startswith("https://"):
            return f'<a href="{url}" color="#0066CC"><u>{label}</u></a>'
        else:
            return f"<u>{label}</u>"

    text = re.sub(r"\[(.*?)\]\((.*?)\)", link_replacer, text)
    # Escape standalone & if not already XML entity
    text = re.sub(r"&(?![a-zA-Z#0-9]+;)", "&amp;", text)
    return text.strip()


def parse_md_to_pdf(md_path, pdf_path):
    with open(md_path, encoding="utf-8") as f:
        lines = f.readlines()

    doc = SimpleDocTemplate(pdf_path, pagesize=letter, leftMargin=54, rightMargin=54, topMargin=54, bottomMargin=54)

    styles = getSampleStyleSheet()

    # Define custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=28,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=15,
        alignment=1,
    )

    h1_style = ParagraphStyle(
        "Header1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=20,
        textColor=colors.HexColor("#1A365D"),
        spaceBefore=16,
        spaceAfter=8,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        "Header2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True,
    )

    h3_style = ParagraphStyle(
        "Header3",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#2D3748"),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=6,
    )

    bullet_style = ParagraphStyle(
        "BulletText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#2D3748"),
        leftIndent=15,
        spaceAfter=4,
    )

    blockquote_style = ParagraphStyle(
        "BlockquoteText",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#4A5568"),
        leftIndent=20,
        rightIndent=20,
        spaceBefore=6,
        spaceAfter=6,
    )

    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#2D3748"),
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=12,
        textColor=colors.white,
    )

    code_style = ParagraphStyle(
        "CodeBlock",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor("#2D3748"),
        leftIndent=10,
        spaceAfter=6,
    )

    story = []
    in_code_block = False
    code_lines = []
    in_table = False
    table_data = []

    i = 0
    while i < len(lines):
        line = lines[i].rstrip("\r\n")

        # Code block handling
        if line.startswith("```"):
            if in_code_block:
                # Close code block
                code_text = "<br/>".join([clean_markdown_inline(c).replace(" ", "&nbsp;") for c in code_lines])
                p = Paragraph(code_text, code_style)
                # Wrap code block in styled box table
                t = Table([[p]], colWidths=[504])
                t.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F7FAFC")),
                            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                            ("PADDING", (0, 0), (-1, -1), 8),
                        ]
                    )
                )
                story.append(t)
                story.append(Spacer(1, 6))
                code_lines = []
                in_code_block = False
            else:
                in_code_block = True
            i += 1
            continue

        if in_code_block:
            code_lines.append(line)
            i += 1
            continue

        # Table handling
        if line.startswith("|"):
            # Check if separator line
            if re.match(r"\|[\s\:\-]+\|", line):
                i += 1
                continue
            row_cells = [clean_markdown_inline(c.strip()) for c in line.split("|")[1:-1]]
            if row_cells:
                table_data.append(row_cells)
            in_table = True
            i += 1
            continue
        elif in_table:
            # End of table encountered
            if table_data:
                formatted_table_data = []
                for row_idx, row in enumerate(table_data):
                    formatted_row = []
                    for cell in row:
                        if row_idx == 0:
                            formatted_row.append(Paragraph(cell, table_header_style))
                        else:
                            formatted_row.append(Paragraph(cell, table_cell_style))
                    formatted_table_data.append(formatted_row)

                # Determine col widths dynamically
                num_cols = max(len(r) for r in formatted_table_data)
                col_width = 504.0 / num_cols
                t = Table(formatted_table_data, colWidths=[col_width] * num_cols)
                t.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1A365D")),
                            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
                            (
                                "ROWBACKGROUNDS",
                                (0, 1),
                                (-1, -1),
                                [colors.white, colors.HexColor("#F7FAFC")],
                            ),
                            ("TOPPADDING", (0, 0), (-1, -1), 6),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                        ]
                    )
                )
                story.append(t)
                story.append(Spacer(1, 8))
            table_data = []
            in_table = False

        # Blank line
        if not line.strip():
            i += 1
            continue

        # Horizontal Rule
        if line.strip() in ["---", "***", "___"]:
            story.append(
                HRFlowable(
                    width="100%",
                    thickness=1,
                    color=colors.HexColor("#E2E8F0"),
                    spaceAfter=10,
                    spaceBefore=10,
                )
            )
            i += 1
            continue

        # Blockquote
        if line.startswith(">"):
            text = clean_markdown_inline(line.lstrip(">").strip())
            story.append(Paragraph(text, blockquote_style))
            story.append(Spacer(1, 4))
            i += 1
            continue

        # Headers
        if line.startswith("# "):
            text = clean_markdown_inline(line[2:])
            if not story:
                # Main Title
                story.append(Spacer(1, 15))
                story.append(Paragraph(text, title_style))
                story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#2B6CB0"), spaceAfter=15))
            else:
                story.append(Paragraph(text, h1_style))
            i += 1
            continue

        if line.startswith("## "):
            text = clean_markdown_inline(line[3:])
            story.append(Paragraph(text, h1_style))
            i += 1
            continue

        if line.startswith("### "):
            text = clean_markdown_inline(line[4:])
            story.append(Paragraph(text, h2_style))
            i += 1
            continue

        if line.startswith("#### "):
            text = clean_markdown_inline(line[5:])
            story.append(Paragraph(text, h3_style))
            i += 1
            continue

        # Bullet lists
        if line.strip().startswith("- ") or line.strip().startswith("* "):
            text = clean_markdown_inline(line.strip()[2:])
            bullet_p = Paragraph(f"• &nbsp; {text}", bullet_style)
            story.append(bullet_p)
            i += 1
            continue

        # Numbered lists
        num_match = re.match(r"^(\d+)\.\s+(.*)", line.strip())
        if num_match:
            num = num_match.group(1)
            text = clean_markdown_inline(num_match.group(2))
            num_p = Paragraph(f"<b>{num}.</b> &nbsp; {text}", bullet_style)
            story.append(num_p)
            i += 1
            continue

        # Normal Paragraph
        text = clean_markdown_inline(line)
        story.append(Paragraph(text, body_style))
        i += 1

    # Handle remaining table if file ends with table
    if in_table and table_data:
        formatted_table_data = []
        for row_idx, row in enumerate(table_data):
            formatted_row = []
            for cell in row:
                if row_idx == 0:
                    formatted_row.append(Paragraph(cell, table_header_style))
                else:
                    formatted_row.append(Paragraph(cell, table_cell_style))
            formatted_table_data.append(formatted_row)
        num_cols = max(len(r) for r in formatted_table_data)
        col_width = 504.0 / num_cols
        t = Table(formatted_table_data, colWidths=[col_width] * num_cols)
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1A365D")),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#F7FAFC")],
                    ),
                ]
            )
        )
        story.append(t)

    # Build PDF
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated PDF: {pdf_path}")


if __name__ == "__main__":
    md_file = r"c:\Users\sachi\OneDrive\Documents\ASTRONOVA\REPORT.md"
    pdf_file = r"c:\Users\sachi\OneDrive\Documents\ASTRONOVA\REPORT.pdf"
    parse_md_to_pdf(md_file, pdf_file)
