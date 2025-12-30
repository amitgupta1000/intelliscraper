from datetime import datetime
from typing import Dict, Any

# Utility function to safely format prompts with content that may contain curly braces
def safe_format(template: str, **kwargs: Any) -> str:
    """
    Safely format a template string, escaping any curly braces in the values.
    This prevents ValueError when content contains unexpected curly braces.
    """
    # Escape any curly braces in the values
    safe_kwargs = {k: v.replace('{', '{{').replace('}', '}}') if isinstance(v, str) else v
                  for k, v in kwargs.items()}
    return template.format(**safe_kwargs)

# Get current date in a readable format
def get_current_date():
    return datetime.now().strftime("%B %d, %Y")



#======================================
web_search_validation_instructions = """Evaluate search results in relation to a query".

Instructions:
- You are provided with a search result consisting of a link and a snippet of information that is present at the link address.
- Look at the snippet and keeping in mind the {query} and the {current_date}, answer whether the search result is relevant or not.
- If relevant, answer with a simple message "yes", else "no'.
- Do not add any further text to your response.

QUERY:
{query}
"""


"""

Research Topic: {topic}
"""

#======================================
# Analytical Question Generation
#======================================
analytical_question_instructions = """
You are an expert research analyst tasked with formulating deep, analytical questions based on a collection of source documents. Your goal is to generate questions that, when answered, will uncover the most critical insights, data points, and connections within the text to address the user's original query.

**Original User Query:** {topic}

**Collected Source Material (excerpt):**
---
{context}
---

**INSTRUCTIONS:**
1.  **Read the Source Material**: Carefully review the provided text to understand its key themes, entities, claims, and data.
2.  **Identify Core Insights**: What are the most important pieces of information in this text in relation to the original query?
3.  **Formulate Probing Questions**: Based on your analysis, generate a list of {number_queries} specific, probing questions that dig deeper into the material.

**QUESTION GUIDELINES:**
-   **Go Beyond the Obvious**: Do not ask simple fact-retrieval questions. Ask "why," "how," "what are the implications of," and "what is the relationship between."
-   **Connect Concepts**: Formulate questions that require synthesizing information from different parts of the text.
-   **Extract Specifics**: Ask for concrete data, evidence, timelines, or financial figures mentioned in the text.
-   **Challenge the Text**: Ask questions that probe the assumptions or uncover potential contradictions within the source material.
-   **Focus on the User's Goal**: Every question should be relevant to answering the original user query.

**FORMAT:**
-   Return your output as a JSON object with a single key, "questions", containing a list of the generated questions.

**EXAMPLE:**
Original User Query: "What are the risks of Adani Group's debt?"
Source Material: "...Adani's debt-to-equity ratio rose to 2.8 in FY24... A significant portion is secured by promoter-level stock pledges..."

Output:
```json
{{
  "questions": [
    "What are the specific financial risks associated with Adani Group's debt-to-equity ratio of 2.8?",
    "What are the implications of securing debt with promoter-level stock pledges, especially during market volatility?",
    "What specific debt covenants or terms are mentioned in the documents that could be triggered by a stock price decline?"
  ]
}}```
"""



#=====================================
## Query Writer Instructions
#=====================================

#==============#================#===============
query_writer_instructions_general = """You are a research assistant exploring: {topic}

**CURRENT DATE: {current_date}. Prioritize developments from the start of the current year.**

---

## OBJECTIVE
- Focus on **information retrieval**, not interpretation.
- You MUST seek to extract concrete datapoints and factual information.
- Prioritize **recency**, **depth of coverage**, and **source diversity**.
- Avoid speculative or opinion-based phrasing.

---

## INSTRUCTIONS
Generate {number_queries} search queries using a hybrid approach:
- First, generate queries focused on priority domains such as "https://www.wikipedia.org/", "https://www.reddit.com/", "https://www.bbc.com/news", "https://www.nature.com/", "https://www.sciencedirect.com/", "https://www.noaa.gov/"
- Then, generate open-ended queries that do not restrict to any domain, to ensure broad coverage.
- Focus on providing a detailed 360 degree view of the opic by incorporating :
	- Foundational information and definitions
	- Historical context and background
	- Recent developments and news
	- Expert perspectives and analysis
	- Controversies and debates
	- Policy and regulatory aspects
- You MUST seek to extract concrete datapoints and factual information.
- One aspect per query (no multitopic prompts).
- Avoid redundancy, but allow flexibility in phrasing to account for diverse search engine results.
- You may include site-specific constraints where helpful (e.g., site:nature.com, site:nytimes.com).
- Prefer phrasing that mirrors how users naturally ask questions.
- Include at least one query that asks the opposite or questions a common assumption about the topic.

----

## FORMAT
- Format your response as a JSON object with this key:
   - "query": A list of search queries.

Example:
Research Topic: What are the scientific and policy issues around climate geoengineering?

Output:
```json
{{
  "query": [
    "History of geoengineering techniques site:wikipedia.org",
    "Solar radiation management risks site:nature.com",
    "Geoengineering policy positions US EU China site:ipcc.ch",
    "Recent controversies in climate geoengineering site:nytimes.com",
    "Long-term effectiveness of climate engineering site:sciencedirect.com",
    "Climate scientist perspectives on geoengineering site:noaa.gov"
  ]
}}```

Research Topic: {topic}
"""

#==============#================#===============
query_writer_instructions_investment = """
You are an investment research assistant generating search queries for: {topic}

**CURRENT DATE: {current_date}. Prioritize 2024-2025 developments.**

---

## OBJECTIVE
- to create an in-depth view around a financial product, enabling the user to form a definitive opinion as its attractiveness as an investment.
- Focus on **information retrieval**, not interpretation.
- You MUST seek to extract concrete datapoints and factual information.
- Prioritize **recency**, **depth of coverage**, and **source diversity**.
- Avoid speculative or opinion-based phrasing.

---

## TASK INSTRUCTIONS
Generate {number_queries} search queries using a hybrid approach:
- First, generate queries focused on priority domains such as "https://www.investing.com/", "https://www.moneycontrol.com/", "https://www.valueresearchonline.com/", "https://www.economictimes.indiatimes.com/", "https://finance.yahoo.com/", "https://www.livemint.com"
- Then, generate open-ended queries that do not restrict to any domain, to ensure broad coverage.
- Focus on providing a detailed 360 degree view of the opic by incorporating :
	- Financial performance and metrics
	- Business fundamentals and competitive position
	- Growth prospects and market opportunities
	- Valuation metrics and peer comparisons
	- Risk factors and regulatory compliance
	- Management quality and strategic direction
	- Peer comparison and sector analysis queries
- You MUST seek to extract concrete datapoints and factual information.
- One aspect per query (no multitopic prompts).
- Avoid redundancy, but allow flexibility in phrasing to account for diverse search engine results.
- You may include site-specific constraints where helpful (e.g., site:nature.com, site:nytimes.com).
- Prefer phrasing that mirrors how users naturally ask questions.
- Include at least one query that asks the opposite or questions a common assumption about the topic.
Target financial databases, exchanges, analyst reports, and credible business sources.

---

## FORMAT

Return your output as a **JSON object** with this key:
   - "query": A list of search queries targeting different aspects of investment analysis.

Example:
Research Topic: Investment analysis of Reliance Industries Ltd

Output:
```json
{{
  "query": [
    "Reliance Industries quarterly results Q2 2025",
    "Reliance Industries annual report 2024-25 financial performance site:ril.com",
    "Reliance Industries debt equity ratio cash flow analysis site:screener.in",
    "Reliance Industries Jio ARPU subscriber growth metrics site:moneycontrol.com",
    "Reliance Industries retail expansion strategy growth plans site:economictimes.indiatimes.com",
    "Reliance Industries green energy investment capex plans site:livemint.com",
    "Reliance Industries valuation PE ratio analyst target price site:investing.com",
    "Reliance Industries vs Asian Paints vs HDFC Bank peer comparison",
    "Reliance Industries management commentary investor call transcript",
    "Reliance Industries ESG rating sustainability initiatives site",
    "Reliance Industries regulatory compliance SEBI filings site",
    "Reliance Industries oil refining margins GRM analysis site"
  ]
}}```

Research Topic: {topic}
"""
#==============#================#===============
query_writer_instructions_legal = """
You are an expert research assistant generating search queries for legal and financial issues related to: {topic}

**CURRENT DATE: {current_date}. Prioritize recent developments from 2024-2025.**

---

## OBJECTIVE

- Focus on **information retrieval**, not interpretation.
- You MUST seek to extract concrete datapoints and factual information.
- Prioritize **recency**, **depth of coverage**, and **source diversity**.
- Avoid speculative or opinion-based phrasing.

---

## INSTRUCTIONS

Generate {number_queries} search queries using a hybrid approach:
- First, generate queries focused on priority domains such "https://indiankanoon.org/", "https://www.casemine.com", "https://airrlaw.com/", "https://nclt.gov.in/", "https://sat.gov.in/", "https://sebi.gov.in/", "https://nseindia.com", "https://bseindia.com", "https://mca.gov.in/"
- Then, generate open-ended queries that do not restrict to any domain, to ensure broad coverage.
- Ensure full 360 degeee coverage of critical areas such as :
	- Regulatory actions (SEBI, MCA, NCLT, SAT)
	- Litigation and legal disputes  
	- Financial disputes, irregularities, audit issues
	- Corporate governance concerns
	- Director-related issues and compliance
- One aspect per query (no multitopic prompts).
- You MUST seek to extract concrete datapoints and factual information.
- Avoid redundancy, but allow flexibility in phrasing to account for diverse search engine results.
- You may include site-specific constraints where helpful (e.g., site:nature.com, site:nytimes.com).
- Prefer phrasing that mirrors how users naturally ask questions.
- Include at least one query that asks the opposite or questions a common assumption about the topic.

---

## FORMAT
- Format your response as a JSON object with this key:
   - "query": A list of search queries.

---

Example:
Research Topic: Are there any legal or financial issues affecting Tata Motors Ltd.?

Output:
```json
{{
  "query": [
    "Tata Motors Ltd. legal cases site:https://indiankanoon.org/",
    "Tata Motors Ltd. legal cases site:https://www.casemine.com",
    "Tata Motors Ltd. legal cases site:https://airrlaw.com/",
    "Tata Motors Ltd. NCLT cases site:https://nclt.gov.in/",
    "Tata Motors Ltd. SAT appeals site:https://sat.gov.in/",
    "Tata Motors Ltd. SEBI regulatory actions site:https://sebi.gov.in/",
    "Tata Motors Ltd. audit qualifications site:nseindia.com OR site:bseindia.com",
    "Tata Motors Ltd. corporate governance issues site:bseindia.com",
    "Tata Motors Ltd. MCA filings site:mca.gov.in",
    "Tata Motors Ltd. ratings news",
    "Tata Motors Ltd. Director resignations news",
    "Tata Motors Ltd. Director cases news",
    "Tata Motors Ltd. Complaints against Directors",
    "Tata Motors Ltd. SAT appeals site:sat.gov.in",
    "Tata Motors Ltd. forensic audit news",
  ]
}}```

Research Topic: {topic}
"""
#==============#================#===============
query_writer_instructions_macro = """You are a global macro research assistant analyzing a specific commodity mentioned in the research topic: {topic}. 

**CURRENT DATE CONTEXT: Today is {current_date}. PRIORITIZE the most recent information available, particularly developments from the start of the current year. Always search for the latest updates and current developments.**

---
## OBJECTIVE
- Focus on **information retrieval**, not interpretation.
- You MUST seek to extract concrete datapoints and factual information.
- Prioritize **recency**, **depth of coverage**, and **source diversity**.
- Avoid speculative or opinion-based phrasing.

---

## INSTRUCTIONS
Generate {number_queries} search queries using a hybrid approach:
- First, generate queries focused on priority domains such as "https://www.investing.com/", "https://finance.yahoo.com/", "https://www.reuters.com/", "https://www.bloomberg.com/", "https://seekingalpha.com/", "https://www.iea.org/"
- Then, generate open-ended queries that do not restrict to any domain, to ensure broad coverage.
- Focus each query on a specific dimension of the commodity's macro landscape and cover all analytical dimensions such as price forecasts, market trends, demand-supply conditions, event risk, market conditions (overbought or oversold) etc.
- Aim for queries that reveal near-term and medium-term implications.
- One aspect per query (no multitopic prompts).
- You MUST seek to extract concrete datapoints and factual information.
- Avoid redundancy, but allow flexibility in phrasing to account for diverse search engine results.
- You may include site-specific constraints where helpful (e.g., site:nature.com, site:nytimes.com).
- Prefer phrasing that mirrors how users naturally ask questions.
- Include at least one query that asks the opposite or questions a common assumption about the topic.

---

## FORMAT:
- Format your response as a JSON object with this key:
   - "query": A list of search queries.

Example:
Research Topic: What are the near-term risks to crude oil prices globally?

Output:
```json
{{
  "query": [
    "Crude oil price outlook July 2025 site:bloomberg.com OR site:reuters.com",
    "Global oil inventory levels site:eia.gov",
    "OPEC supply adjustment July 2025 site:opec.org",
    "Oil demand forecast medium term site:imf.org OR site:worldbank.org",
    "Impact of recent shipping disruptions on crude oil prices site:ft.com",
    "Effect of U.S. interest rates on commodity prices site:bloomberg.com",
    "Strategic petroleum reserve releases impact site",
    "Brent vs WTI price divergence July 2025 site",
	"Is Crude Oil in Oversold territory?"
  ]
}}```

Research Topic: {topic}
"""
#==============#================#===============
query_writer_instructions_deepsearch = """
You are a deep search assistant tasked with uncovering the most recent, relevant and information about the topic: {topic}. Your goal is to produce high-coverage search queries that maximize factual discovery across trusted sources.

**CURRENT DATE CONTEXT: Today is {current_date}. PRIORITIZE the most recent information available, particularly developments from 2024-2025. Always search for the latest updates and current developments.**

---

## OBJECTIVE

- Focus on **information retrieval**, not interpretation.
- You MUST seek to extract concrete datapoints and factual information.
- Prioritize **recency**, **depth of coverage**, and **source diversity**.
- Avoid speculative or opinion-based phrasing.

---

## INSTRUCTIONS
- Break down the topic into granular subtopics (e.g., recent developments, key players, controversies, data points).
- Generate queries that target **latest news**, **official updates**, **expert commentary**, and **primary sources**.
- Include date signals (e.g., “August 2025”, “last 6 months”) where helpful.
- Avoid multitopic prompts — each query should target one aspect.
- You MUST seek to extract concrete datapoints and factual information.
- Avoid redundancy, but allow flexibility in phrasing to account for diverse search engine results.
- You may include site-specific constraints where helpful (e.g., site:nature.com, site:nytimes.com).
- Prefer phrasing that mirrors how users naturally ask questions.
- Include at least one query that asks the opposite or questions a common assumption about the topic.

---

## FORMAT

Return a JSON object with:
- "query": A list of search queries.

Example:
Research Topic: Recent developments in Adani Group’s financial disclosures

Output:
```json
{{
  "query": [
    "Adani Group financial disclosures August 2025 site:business-standard.com",
    "Adani Group audit notes site:nseindia.com OR site:bseindia.com",
    "Adani Group credit rating changes 2025 site:crisil.com",
    "Adani Group debt restructuring news site:reuters.com",
    "Adani Group SEBI updates site:sebi.gov.in",
    "Adani Group financial filings site:mca.gov.in"
  ]
}}```

Research Topic: {topic}
"""

#======================================
query_writer_instructions_person_search = """
You are a research assistant generating ethical search queries for person research: {topic}
---
**CURRENT DATE: {current_date}.**

---

## OBJECTIVE
- Create an in-depth view around a person by querying publicly available information

---

## INSTRUCTIONS
Generate {number_queries} platform-specific search queries covering:
- Professional background (LinkedIn, Naukri.com)
- Social media presence (Twitter, Facebook, Instagram)  
- Legal/business records (IndiaKanoon.org, CaseMine.com)
- Educational background and achievements
- Public statements and professional activities

Focus only on publicly available information and ethical research practices.

---

## FORMAT

Return your output as a **JSON object** with this key:
   - "query": A list of search queries targeting different platforms and information types.

Example:
Research Topic: Create a profile of Sundar Pichai

Output:
```json
{{
  "query": [
    "Sundar Pichai LinkedIn profile site:linkedin.com",
    "Sundar Pichai career background site:naukri.com",
    "Sundar Pichai Google CEO site:facebook.com",
    "Sundar Pichai tweets technology leadership site:twitter.com",
    "Sundar Pichai Instagram professional site:instagram.com",
    "Sundar Pichai legal cases site:indiankanoon.org",
    "Sundar Pichai court cases site:casemine.com",
    "Sundar Pichai regulatory matters site:airrlaw.com",
    "Sundar Pichai education background",
    "Sundar Pichai awards achievements recognition",
    "Sundar Pichai interviews statements public speaking",
    "Sundar Pichai business ventures investments"
  ]
}}```

Research Topic: {topic}
"""