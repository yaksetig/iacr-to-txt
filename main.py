from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
import requests
import PyPDF2
import pdfplumber
import fitz  # PyMuPDF
import re
import io
from typing import Dict, List, Optional, Any
import logging
from urllib.parse import urlparse
import asyncio
import aiohttp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="IACR Cryptography Paper Extractor", version="1.0.0")

class PDFRequest(BaseModel):
    url: HttpUrl
    extraction_method: str = "comprehensive"  # "basic", "detailed", "comprehensive"

class PaperContent(BaseModel):
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    abstract: Optional[str] = None
    introduction: Optional[str] = None
    sections: Optional[Dict[str, str]] = None
    conclusion: Optional[str] = None
    references: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    full_text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class PDFExtractor:
    def __init__(self):
        self.section_patterns = [
            r'^\d+\.?\s+(.+)$',  # 1. Section or 1 Section
            r'^[IVX]+\.?\s+(.+)$',  # Roman numerals
            r'^[A-Z][A-Z\s]+$',  # ALL CAPS sections
            r'^\w+\s*\n',  # Single word followed by newline
        ]
        
        self.common_sections = {
            'abstract': ['abstract', 'summary'],
            'introduction': ['introduction', 'intro'],
            'background': ['background', 'preliminaries', 'related work'],
            'methodology': ['methodology', 'method', 'approach', 'construction'],
            'results': ['results', 'analysis', 'evaluation'],
            'conclusion': ['conclusion', 'conclusions', 'summary'],
            'references': ['references', 'bibliography'],
            'acknowledgments': ['acknowledgments', 'acknowledgements']
        }

    async def download_pdf(self, url: str) -> bytes:
        """Download PDF from URL with proper error handling"""
        try:
            # Handle IACR specific URLs
            if 'iacr.org' in url and '/eprint/' in url:
                # Convert to direct PDF URL if needed
                if not url.endswith('.pdf'):
                    url = url.rstrip('/') + '.pdf'
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=30) as response:
                    if response.status != 200:
                        raise HTTPException(status_code=400, detail=f"Failed to download PDF: HTTP {response.status}")
                    
                    content = await response.read()
                    if len(content) == 0:
                        raise HTTPException(status_code=400, detail="Downloaded file is empty")
                    
                    return content
                    
        except Exception as e:
            logger.error(f"Error downloading PDF: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error downloading PDF: {str(e)}")

    def extract_with_pypdf2(self, pdf_bytes: bytes) -> Dict[str, Any]:
        """Extract text using PyPDF2 - basic extraction"""
        try:
            pdf_file = io.BytesIO(pdf_bytes)
            reader = PyPDF2.PdfReader(pdf_file)
            
            text = ""
            metadata = {}
            
            # Extract metadata
            if reader.metadata:
                metadata = {
                    'title': reader.metadata.get('/Title', ''),
                    'author': reader.metadata.get('/Author', ''),
                    'subject': reader.metadata.get('/Subject', ''),
                    'creator': reader.metadata.get('/Creator', ''),
                    'pages': len(reader.pages)
                }
            
            # Extract text from all pages
            for page in reader.pages:
                text += page.extract_text() + "\n"
            
            return {'text': text, 'metadata': metadata}
            
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed: {str(e)}")
            return {'text': '', 'metadata': {}}

    def extract_with_pdfplumber(self, pdf_bytes: bytes) -> Dict[str, Any]:
        """Extract text using pdfplumber - better for tables and layout"""
        try:
            pdf_file = io.BytesIO(pdf_bytes)
            text = ""
            tables = []
            
            with pdfplumber.open(pdf_file) as pdf:
                metadata = {
                    'pages': len(pdf.pages),
                    'metadata': pdf.metadata or {}
                }
                
                for page in pdf.pages:
                    # Extract text
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                    
                    # Extract tables
                    page_tables = page.extract_tables()
                    if page_tables:
                        tables.extend(page_tables)
            
            return {'text': text, 'metadata': metadata, 'tables': tables}
            
        except Exception as e:
            logger.error(f"pdfplumber extraction failed: {str(e)}")
            return {'text': '', 'metadata': {}}

    def extract_with_pymupdf(self, pdf_bytes: bytes) -> Dict[str, Any]:
        """Extract text using PyMuPDF - most comprehensive"""
        try:
            pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            text = ""
            metadata = pdf_doc.metadata
            toc = pdf_doc.get_toc()  # Table of contents
            
            for page_num in range(pdf_doc.page_count):
                page = pdf_doc[page_num]
                text += page.get_text() + "\n"
            
            pdf_doc.close()
            
            return {
                'text': text, 
                'metadata': metadata, 
                'toc': toc,
                'pages': pdf_doc.page_count
            }
            
        except Exception as e:
            logger.error(f"PyMuPDF extraction failed: {str(e)}")
            return {'text': '', 'metadata': {}}

    def parse_paper_structure(self, text: str) -> PaperContent:
        """Parse the extracted text into structured paper content"""
        content = PaperContent()
        
        # Clean the text
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Extract title (usually first significant line)
        lines = text.split('\n')
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if len(line) > 10 and not line.lower().startswith(('abstract', 'keywords')):
                # Check if it looks like a title
                if not re.match(r'^\d+', line) and len(line.split()) > 2:
                    content.title = line
                    break
        
        # Extract abstract
        abstract_match = re.search(r'abstract[:\n\s]+(.*?)(?=\n\s*\n|\n\s*1\s|\n\s*introduction|\n\s*keywords)', text, re.IGNORECASE | re.DOTALL)
        if abstract_match:
            content.abstract = abstract_match.group(1).strip()
        
        # Extract keywords
        keywords_match = re.search(r'keywords[:\s]+(.*?)(?=\n\s*\n|\n\s*1\s|\n\s*introduction)', text, re.IGNORECASE)
        if keywords_match:
            keywords_text = keywords_match.group(1).strip()
            content.keywords = [kw.strip() for kw in re.split(r'[,;]', keywords_text) if kw.strip()]
        
        # Extract authors (look for patterns after title)
        author_patterns = [
            r'(?:by\s+)?([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s*,\s*[A-Z][a-z]+\s+[A-Z][a-z]+)*)',
            r'([A-Z][a-z]+(?:\s+[A-Z]\.)*\s+[A-Z][a-z]+(?:\s*,\s*[A-Z][a-z]+(?:\s+[A-Z]\.)*\s+[A-Z][a-z]+)*)'
        ]
        
        for pattern in author_patterns:
            author_match = re.search(pattern, text[:1000])  # Look in first 1000 chars
            if author_match:
                authors_text = author_match.group(1)
                content.authors = [author.strip() for author in re.split(r'[,&]|\sand\s', authors_text) if author.strip()]
                break
        
        # Extract sections
        content.sections = self.extract_sections(text)
        
        # Extract introduction and conclusion specifically
        if content.sections:
            for key, value in content.sections.items():
                if 'introduction' in key.lower():
                    content.introduction = value
                elif 'conclusion' in key.lower():
                    content.conclusion = value
        
        # Extract references
        content.references = self.extract_references(text)
        
        # Store full text
        content.full_text = text
        
        return content

    def extract_sections(self, text: str) -> Dict[str, str]:
        """Extract sections from the paper"""
        sections = {}
        
        # Find section headers
        section_pattern = r'\n\s*(\d+(?:\.\d+)*\.?\s+[A-Z][^.\n]*(?:\.[^.\n]*)*)\s*\n'
        matches = list(re.finditer(section_pattern, text, re.MULTILINE))
        
        if not matches:
            # Try alternative patterns for section detection
            section_pattern = r'\n\s*([A-Z][A-Z\s]{3,})\s*\n'
            matches = list(re.finditer(section_pattern, text, re.MULTILINE))
        
        for i, match in enumerate(matches):
            section_title = match.group(1).strip()
            start_pos = match.end()
            
            # Find end position (next section or end of text)
            if i + 1 < len(matches):
                end_pos = matches[i + 1].start()
            else:
                end_pos = len(text)
            
            section_content = text[start_pos:end_pos].strip()
            
            # Clean section content
            section_content = re.sub(r'\n+', ' ', section_content)
            section_content = re.sub(r'\s+', ' ', section_content)
            
            if len(section_content) > 50:  # Only include substantial sections
                sections[section_title] = section_content
        
        return sections

    def extract_references(self, text: str) -> List[str]:
        """Extract references from the paper"""
        references = []
        
        # Find references section
        ref_pattern = r'(?:references|bibliography)\s*\n(.*?)(?:\n\s*\n|\Z)'
        ref_match = re.search(ref_pattern, text, re.IGNORECASE | re.DOTALL)
        
        if ref_match:
            ref_text = ref_match.group(1)
            
            # Split by numbered references
            ref_items = re.split(r'\n\s*\[\d+\]|\n\s*\d+\.', ref_text)
            
            for ref in ref_items:
                ref = ref.strip()
                if len(ref) > 20:  # Filter out short/empty references
                    references.append(ref)
        
        return references

    async def extract_content(self, pdf_bytes: bytes, method: str = "comprehensive") -> PaperContent:
        """Main extraction method that combines different approaches"""
        
        if method == "basic":
            result = self.extract_with_pypdf2(pdf_bytes)
        elif method == "detailed":
            result = self.extract_with_pdfplumber(pdf_bytes)
        else:  # comprehensive
            # Try multiple methods and use the best result
            results = []
            
            pymupdf_result = self.extract_with_pymupdf(pdf_bytes)
            if pymupdf_result['text']:
                results.append(('pymupdf', pymupdf_result))
            
            pdfplumber_result = self.extract_with_pdfplumber(pdf_bytes)
            if pdfplumber_result['text']:
                results.append(('pdfplumber', pdfplumber_result))
            
            pypdf2_result = self.extract_with_pypdf2(pdf_bytes)
            if pypdf2_result['text']:
                results.append(('pypdf2', pypdf2_result))
            
            if not results:
                raise HTTPException(status_code=500, detail="Failed to extract text from PDF")
            
            # Use the result with the most text
            best_result = max(results, key=lambda x: len(x[1]['text']))
            result = best_result[1]
        
        # Parse the structure
        content = self.parse_paper_structure(result['text'])
        content.metadata = result.get('metadata', {})
        
        return content

# Initialize extractor
extractor = PDFExtractor()

@app.get("/")
async def root():
    return {"message": "IACR Cryptography Paper Extractor API", "version": "1.0.0"}

@app.post("/extract", response_model=PaperContent)
async def extract_paper(request: PDFRequest):
    """
    Extract content from an IACR cryptography paper PDF
    
    - **url**: Direct URL to the PDF file
    - **extraction_method**: Method to use (basic, detailed, comprehensive)
    """
    try:
        # Download PDF
        pdf_bytes = await extractor.download_pdf(str(request.url))
        
        # Extract content
        content = await extractor.extract_content(pdf_bytes, request.extraction_method)
        
        return content
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
