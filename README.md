# Voyager Agent: Trip Planner AI Capstone

A production-style trip planning application built with Streamlit that combines live location data, optional retrieval grounding, interactive mapping, persistent state, and an agentic planning workflow.

The app is designed to demonstrate practical AI engineering skills rather than just basic prompting. It pulls real points of interest from OpenStreetMap, can optionally use OpenAI's Responses API with tool calling for itinerary generation and refinement, supports local persistence, visualizes plans on a map, and includes a lightweight feedback loop that improves future recommendations for the same destination.

## Why this project matters

This project showcases several skills that are useful in real AI product development:

- Building an interactive app with Streamlit
- Designing tool-based agent workflows
- Integrating live public APIs
- Constraining model outputs for reliability
- Validating generated content against real data
- Handling retries, caching, and persistence
- Creating useful UX around AI outputs

It is structured as a capstone-style portfolio project with enough complexity to show engineering ability while remaining lightweight enough to run locally.

## Core features

### 1. Live destination and POI discovery
The app uses public OpenStreetMap services to find a destination and collect relevant points of interest:

- Nominatim for geocoding a destination into coordinates
- Overpass API for discovering nearby POIs such as museums, parks, cafes, viewpoints, restaurants, galleries, bars, and historic locations

These POIs are then ranked and used as the allowed set of itinerary locations.

### 2. Optional agentic planning with OpenAI tool calling
When an OpenAI API key is provided, the app can use the Responses API with structured function calling.

The agent can:
- call tools to search for POIs
- retrieve supporting destination context
- reason across multiple steps
- return itinerary JSON that is validated against known POIs

This is the more advanced AI workflow in the project and reflects how production systems often mix models with external tools.

### 3. Free no-key fallback mode
Because API usage costs money, this project also includes a built-in fallback planner.

If no OpenAI API key is entered:
- the app still works
- it fetches live POIs from OpenStreetMap
- it builds a deterministic itinerary from those results
- it renders the itinerary and map normally

This means the project remains fully demoable without paid model access. The OpenAI-powered agent path is implemented and ready, but I was not able to actively use and test that branch with sustained real calls due to funding constraints for API credits.

### 4. Optional Wikivoyage retrieval
The app can optionally retrieve and rank destination information from Wikivoyage using a simple TF-IDF retrieval layer.

This demonstrates a lightweight RAG-style workflow:
- fetch destination guide text
- chunk it
- vectorize it with TF-IDF
- retrieve the most relevant chunks for itinerary support

If retrieval is disabled, unavailable, or blocked by the public endpoint, the planner continues gracefully.

### 5. Interactive map visualization
The generated trip is displayed on an interactive PyDeck map with:

- POI markers
- per-day filtering
- path rendering across the day
- light and dark map modes

This makes the itinerary easier to inspect and turns the project into something more product-like than a plain JSON generator.

### 6. Persistence and feedback
The app stores local state so work is not lost between reruns.

It also includes a simple feedback loop:
- users can upvote or downvote places
- votes are stored locally
- future ranking is slightly adjusted for the same destination

This is a simple but useful example of iterative preference shaping.

## Architecture overview

The application has four main layers:

### UI layer
Built with Streamlit. Handles:
- user inputs
- settings
- itinerary rendering
- map rendering
- refinement actions
- feedback actions

### Data layer
Handles external public API calls to:
- Nominatim
- Overpass
- Wikivoyage

Also handles local file persistence for:
- saved itinerary state
- POI vote history

### Planning layer
Supports two planning modes:

**A. OpenAI agent mode**
- uses OpenAI Responses API
- uses strict tool schemas
- validates generated `poi_id` values against fetched POIs

**B. Fallback mode**
- uses deterministic ranking and allocation logic
- works without any paid API access
- still creates a usable multi-day itinerary

### Validation layer
Ensures:
- JSON output is parseable
- only valid POIs are used
- day-specific regeneration does not accidentally modify other days

## How the OpenAI function calling works

When a valid OpenAI API key is supplied, the app can run an agent loop that calls tools. At a high level:

1. The user provides trip inputs
2. The app sends a structured prompt to the model
3. The model can call functions such as:
   - `find_pois(...)`
   - `get_destination_guide(...)`
4. The app executes those tools in Python
5. Tool results are fed back into the model
6. The model returns final itinerary JSON
7. The app validates that every `poi_id` actually came from the tool outputs

This creates a safer workflow than unconstrained text generation because the model is grounded to real external data and checked before the result is accepted.

## Limitation on active model usage

The OpenAI-based tool-calling workflow is implemented in the codebase, but I could not regularly use that pathway in practice because I do not currently have spare budget for API credits.

For that reason, the project was intentionally designed with a full fallback path so it can still be run, demonstrated, and evaluated without a paid key.

In other words:
- the OpenAI agent integration is present
- the app is still usable without it
- the fallback mode keeps the capstone functional and testable

## Tech stack

- Python
- Streamlit
- OpenAI Python SDK
- Requests
- NumPy
- PyDeck
- scikit-learn

## Project structure

```text
.
├── trip_planner_agent.py
├── requirements.txt
├── .gitignore
└── data/
    ├── voyager_state.json
    └── poi_votes.jsonl
```

The `data/` folder is created during use and stores local state and feedback data.

## Installation

### 1. Create an environment
Using conda:

```bash
conda create -n tripplanner python=3.11 -y
conda activate tripplanner
```

Or using venv on Windows:

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run trip_planner_agent.py
```

If `streamlit` is not recognized:

```bash
python -m streamlit run trip_planner_agent.py
```

## How to use the app

### Free mode
Leave the OpenAI API key field blank.

Then:
1. Enter a destination city
2. Choose trip length, pace, and interests
3. Add constraints or notes
4. Click **Generate itinerary**

The app will:
- geocode the destination
- fetch live POIs
- generate a fallback itinerary
- display the itinerary and map
- allow feedback and refinement

### OpenAI agent mode
Enter a valid OpenAI API key in the sidebar.

Then the app can use model-based planning and tool calling for:
- itinerary generation
- global refinement
- single-day regeneration

## Important runtime notes

### Public API etiquette
The app uses public services, so requests can occasionally be blocked or rate limited.

To reduce issues:
- use a real email in the **User-Agent contact** field
- avoid spamming many repeated requests quickly
- retry after a short pause if needed

> **Note:** To use the live OpenStreetMap geocoding and POI features reliably, enter a real email address in the app’s **User-Agent contact** field, as public endpoints may reject placeholder or generic identifiers.

### Optional guide retrieval
Wikivoyage retrieval may sometimes return no content or reject requests. The app is built to continue even when that happens.

## Reliability choices in the app

The project includes several practical reliability measures:

- caching for repeated external lookups
- output validation for model JSON
- POI ID validation against tool results
- local autosave
- visible execution trace
- graceful fallback behavior
- safe refinement checks

These choices are important because AI applications become much more useful when they are constrained, inspectable, and resilient.

## Potential future improvements

Some natural extensions would be:

- travel time estimation between POIs
- hotel and restaurant budgeting
- multi-city planning
- stronger retrieval with embeddings
- saved trip libraries with a real database
- user accounts and shared itineraries
- export to PDF or calendar

## Summary

Voyager Agent is a practical AI capstone project that blends:
- real external data
- optional tool-calling AI
- retrieval grounding
- interactive visualization
- persistence
- user feedback

It is intentionally designed to remain useful even without paid model access, while still showing how a more advanced agentic workflow would operate in a production-style application.
