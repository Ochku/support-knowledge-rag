# Architecture

Four views: **offline ingestion**, **online query**, **evaluation loop**, and **how it would sit at FOSS**.
The diagrams are Mermaid: they render on GitHub, and in VS Code with a Mermaid preview extension. A plain-text summary is at the bottom.

## 1. Offline: ingestion and indexing
```mermaid
flowchart LR
    subgraph RAW["data/raw (public FOSS docs)"]
        P["Product pages<br/>71 EN + 8 DE"]
        S["Service & support pages<br/>11"]
        F["PDF papers<br/>13 (137 pages)"]
        C["Product catalogue JSON<br/>71 products, structured"]
    end
    M[("data/sources.tsv<br/>url · kind · lang · industries")]

    P --> CL["Parse & clean<br/>strip nav, cookie banner,<br/>quote forms, country lists"]
    S --> CL
    F --> PX["PDF text per page<br/>flag garbage pages"]
    M -. metadata join .-> CL
    M -. metadata join .-> PX
    CL --> CH["Chunk<br/>heading-aware, ~300-500 tokens<br/>title + heading prepended"]
    PX --> CH
    C --> CAT["Catalogue records<br/>(analysis time, principle,<br/>parameters, industries)"]

    CH --> DE["Dense embed<br/>multilingual MiniLM / e5-large<br/>(fastembed, local)"]
    CH --> SP["Sparse embed<br/>BM25 (fastembed)"]
    DE --> Q[("Qdrant collection<br/>named vectors: dense + bm25<br/>payload: url, kind, lang,<br/>industries, product, heading, page")]
    SP --> Q
    CAT --> Q
    CAT -. optional .-> LT["Lookup table for<br/>lookup_product tool"]
```

## 2. Online: answering a question
```mermaid
sequenceDiagram
    autonumber
    actor U as Support engineer / customer
    participant A as Answer agent (Claude)
    participant T as search_docs tool
    participant Q as Qdrant (hybrid)
    participant L as lookup_product tool (stretch)

    U->>A: Question (any language)
    A->>T: search_docs(query, industry?, lang?)
    T->>Q: prefetch dense top-20 + bm25 top-20
    Q-->>T: RRF-fused top-k chunks + payload
    T-->>A: Numbered passages [1..k] with url/page
    opt specs question
        A->>L: lookup_product("MilkoScan FT3")
        L-->>A: analysis time, principle, parameters
    end
    alt sources answer the question
        A-->>U: Answer in user's language + [n] citations
    else not covered by the docs
        A-->>U: "I can't confirm this from the documentation, contact your local FOSS support" (escalate)
    end
    Note over A,Q: Logged per question: tokens, cost, latency,<br/>tool calls, retrieved chunk IDs
```

## 3. Evaluation loop
```mermaid
flowchart LR
    E[("eval/cases.jsonl<br/>35 cases, written BEFORE tuning")] --> R["Retrieval run<br/>per search mode"]
    R --> RM["hit@1/3/5 · recall@5 · MRR<br/>overall + per category"]
    E --> AN["Answer run<br/>(agent + tools)"]
    AN --> DC["Deterministic checks<br/>citation validity · language match<br/>refusal on unanswerables"]
    AN --> J["LLM judge<br/>eval/rubric.md<br/>structured verdict"]
    J --> CAL["Calibration<br/>10 hand-labelled cases"]
    RM --> RES[("results/run-*.json<br/>config + metrics + per-case")]
    DC --> RES
    J --> RES
    RES --> TAB["README results table<br/>config A vs config B"]
```

## 4. How this maps to FOSS (hypothesis, based on the posting)
The posting says FOSS has a **Boost.ai customer chatbot**, "**AI agents that handle specialist retrieval and lookups**", models on **Azure OpenAI / AI Foundry**, **Qdrant**, and integrations with SharePoint, Salesforce and the company apps. This prototype is the "specialist retrieval" piece.

```mermaid
flowchart LR
    CU["Customer"] --> BA["Boost.ai chatbot<br/>(flows, intents)"]
    SE["Support engineer"] --> APP["Internal tools<br/>(Salesforce, SharePoint, apps)"]
    BA -- "needs product/method knowledge" --> AG["Coordinator agent"]
    APP --> AG
    AG --> R1["Retrieval specialist<br/>(this prototype)"]
    AG --> R2["Lookup specialist<br/>(catalogue, instrument data)"]
    R1 --> QD[("Qdrant")]
    R2 --> SYS[("Product master data,<br/>FOSS IQX device data?")]
    AG --> LLM["Azure OpenAI / AI Foundry<br/>(here: Claude, behind an interface)"]
    AG -- "not answerable" --> ESC["Escalate to local<br/>FOSS support"]
    SE -. "verdicts / feedback" .-> EV[("Eval set grows from<br/>real support questions")]
    EV -.-> AG
```
**Talking point:** "In my prototype the model client and the embedding model sit behind one interface, so moving to Azure OpenAI is a configuration change, and Qdrant is the same as yours. The part I can't build without you is the evaluation set: it should come from your support engineers' real questions."

---

## Plain-text summary
```
OFFLINE  data/raw ─► parse+clean ─► chunk(+title/heading) ─► dense(e5/MiniLM) + sparse(BM25) ─► Qdrant(payload: url,kind,lang,industries,page)
ONLINE   question ─► Claude ─tool─► search_docs ─► Qdrant hybrid (prefetch dense+bm25 → RRF) ─► top-k ─► answer with [n] citations | escalate
EVAL     cases.jsonl ─► retrieval metrics + answers ─► deterministic checks + LLM judge(rubric) ─► results/*.json ─► README table
```

## Component choices (see `decisions.md` for the why)
| Component | Choice | Swap-able to |
|---|---|---|
| Vector DB | Qdrant 1.20 (Docker or embedded) | (FOSS uses Qdrant) |
| Dense embeddings | fastembed: `paraphrase-multilingual-MiniLM-L12-v2` → `multilingual-e5-large` | Azure OpenAI embeddings |
| Sparse | fastembed `Qdrant/bm25` (IDF in Qdrant) | SPLADE / miniCOIL |
| Fusion | Qdrant Query API, RRF | DBSF, weighted |
| LLM | Claude Opus 5.5 via the `anthropic` SDK, manual tool loop | Azure OpenAI / Foundry model |
| Judge | Claude, structured output, rubric | Human labels from support engineers |
