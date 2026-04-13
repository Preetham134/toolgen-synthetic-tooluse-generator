# ToolGen Design

## 1. Problem Framing

ToolGen is built to generate offline synthetic tool-use conversations for evaluation, prototyping, and pipeline testing. The project targets the common need for multi-step conversations that call tools in a grounded sequence, while avoiding dependence on live APIs or expensive online generation loops.

Synthetic tool-use conversations matter because they let a team:

- exercise orchestration logic without production tool access
- create deterministic evaluation fixtures
- compare design variants such as steering on vs off
- inspect failure modes in chain planning, grounding, repair, and scoring

In this repository, "offline generation" means the conversation generator does not require live tool backends. Tool results are created by a mock executor, and the judge is heuristic-first.

## 2. System Architecture

The full pipeline is:

`raw tools -> registry -> graph -> sampler -> planner -> generator/orchestrator -> executor/state -> validators -> repair -> evaluation`

Module responsibilities:

- `registry`
  Loads raw JSON, handles schema variation, and normalizes records into a clean internal endpoint format.
- `graph`
  Converts normalized endpoints into nodes and adds directed scored edges between likely tool transitions.
- `sampler`
  Chooses chains from the graph with light semantic preferences and optional cross-conversation steering.
- `planner`
  Builds a deterministic user goal, optional clarification, and starter arguments.
- `generator/orchestrator`
  Coordinates chain sampling, argument construction, tool execution, validation, and repair into a `ConversationRecord`.
- `executor/state`
  Simulates tool execution and stores reusable IDs, entities, slots, counters, and tool-output history.
- `validators`
  Checks structure, registry membership, required params, and entity-type-aware grounding.
- `repair`
  Applies targeted deterministic fixes instead of regenerating whole conversations.
- `evaluation`
  Scores conversations and computes aggregate dataset metrics.

This modular architecture was chosen because each stage can be tested independently, while still composing into an end-to-end reproducible pipeline.

### Agent Roles And Communication Protocol

Although the implementation is module-based rather than a runtime multi-agent system, the architecture maps naturally onto agent-style roles:

- Planner agent
  - represented by the planner templates that turn a sampled chain into a user goal, optional clarification, and initial arguments
- Sampler agent
  - represented by the graph sampler that chooses an endpoint chain from the processed graph
- Generator/Orchestrator agent
  - represented by the orchestrator that coordinates planning, argument filling, execution, validation, repair, and record assembly
- Executor agent
  - represented by the mock executor and conversation state, which simulate tool outputs and preserve reusable values across steps
- Validator agent
  - represented by the deterministic validators for structure, required params, registry membership, and grounding
- Repair agent
  - represented by the deterministic repair module that patches targeted failures without redesigning the whole conversation
- Judge/Evaluator agent
  - represented by the heuristic judge and dataset-metrics layer

The communication protocol between these roles is fixed and simple:

1. Registry and graph artifacts are built first from raw tool definitions.
2. The sampler returns a chain of endpoints.
3. The planner turns that chain into a user goal, optional clarification, and initial arguments.
4. The orchestrator executes the chain step by step.
5. The executor returns mock tool outputs and updates conversation state.
6. The validator returns `passed/issues` results for the assembled record.
7. The repair stage consumes those issues and may revise the record with targeted fixes.
8. The evaluator scores the final records and writes summary metrics.

This protocol is intentionally concrete and linear. It reduces ambiguity, keeps each role testable, and makes failure analysis easier because every stage has a clear input and output boundary.

## 3. Tool Registry Design

Raw tool catalogs are often inconsistent. The registry layer exists to normalize those differences into a stable endpoint schema used everywhere else in the system.

Normalized endpoint fields include:

- `endpoint_id`
- `tool_name`
- `api_name`
- `category`
- `description`
- `input_params`
- `required_params`
- `output_schema`
- `tags`
- `source_tool_id`

The loader handles variation in raw fields such as:

- tool name from `tool_name`, `name`, `tool`, `api_name`
- api name from `api_name`, `endpoint`, `name`, `action`
- category from `category`, `domain`, `group`
- parameter lists from `input_params`, `parameters`, `params`, or `input_schema.properties`

The purpose of this step is not to fully understand arbitrary API specs. It is to create a consistent internal representation that later stages can rely on.

## 4. Graph Design

Each normalized endpoint becomes a graph node. Directed edges are added when one endpoint plausibly leads to another.

Earlier in development, graph construction was more permissive. That version created too many weak cross-domain or cross-family transitions. The final version tightens edge creation substantially.

Current edge policy:

- same family is strong
  - hotel -> hotel
  - flight -> flight
  - product -> product
  - restaurant -> restaurant
- compatible family is moderate
  - product <-> order
  - hotel <-> flight is allowed but weaker
- same tool name adds support
- same category is only a weak bonus
- entity-ID overlap is strong only when the families match or are explicitly compatible
- generic overlaps like `city` or `date` are weak and not enough on their own to justify a strong edge
- `search_to_action` is only counted when the source and target families are same or compatible

This produces a graph that is still flexible, but less likely to support obviously odd chains such as hotel search flowing into restaurant booking or product search flowing into flight booking.

Using the richer graph artifact, the final tightening pass reduced the graph from `51` edges to `28` edges, which cut weak cross-domain transitions and improved chain locality.

Using the saved rich sample artifact:

- `sample_tools_rich_graph.json`
  - `10` nodes
  - `28` edges after the final tightening pass

## 5. Sampling And Generation Design

Sampling starts from the graph and tries to produce a chain of endpoints with the requested length. The sampler now prefers:

- search-first starts when possible
- same-family continuation
- compatible-family fallback when same-family is unavailable
- same-domain fallback only when stronger options are missing

This keeps common patterns such as:

- `search -> details`
- `search -> book`
- `search -> details -> book`
- `search -> details -> order`

The planner is template-based and deterministic. It creates:

- a simple user goal
- optional clarification when fields like `city`, `date`, or `category` are required
- initial arguments such as `city = Paris` or `category = electronics`

The orchestrator then:

- samples the chain
- builds arguments per step
- executes tools through the mock executor
- appends assistant/tool messages
- validates the conversation
- applies deterministic repair if needed

## Prompt Design

The current system does not rely on a large free-form prompting layer, but it still contains prompt-like template components.

Template-driven components include:

- user-goal templates
  - examples such as "Find me a hotel in Paris", "Search for a product", or "Look up restaurant options"
- clarification prompts
  - simple assistant questions such as "What city would you like me to use?"
- assistant tool-call wording
  - consistent phrases like "I'll use `endpoint_name` to help with this request."
- final assistant response templates
  - short completion messages such as "I found some options and completed the requested steps."
- heuristic judge prompt substitute
  - instead of prompting an LLM, the evaluator uses structured rule-based scoring logic over the final record

These pieces were kept simple and deterministic for reproducibility. The project is designed to make repeated runs easy to compare, and template-based generation is much easier to test and debug than unconstrained free-form generation.

Structured generation was preferred because it improves:

- reproducibility
- testability
- cost control
- failure localization

One lesson from earlier iterations was that looser generation behavior produced booking/details calls too early, which made chains less natural and harder to validate. Another lesson was that generic repair or fallback IDs made records structurally passable but semantically less coherent. The search-first bias and tighter template constraints improved quality by reducing those failure modes.

## 6. Offline Execution And State

The executor is intentionally simple and offline.

It supports:

- search/list/find
- details/get
- book/reserve/order/buy/create
- update/modify
- cancel/delete
- fallback echo behavior

The `ConversationState` stores:

- entities found so far
- slots copied from arguments
- tool-output history
- counters for deterministic IDs like `hotel_001`

This lets later tool calls reuse earlier results, which is necessary for grounded chains such as:

- hotel search -> hotel details -> hotel book
- product search -> product details -> order create

## Metadata Schema

Generated records carry metadata for reproducibility, filtering, debugging, and dataset analysis.

Key fields include:

- `seed`
  - used to make runs reproducible and trace deterministic sampling/planning decisions
- `chain_length`
  - used for filtering and analyzing single-step versus multi-step behavior
- `chain_pattern`
  - a compact summary such as `search->details->book` or `search->order` for debugging and quality inspection
- `tools_used`
  - records which tools appeared in the chain for usage analysis and steering diagnostics
- `categories`
  - records the domain/category path for diversity analysis and locality checks
- `had_clarification`
  - indicates whether the conversation included a clarification exchange
- `initial_arguments`
  - preserves the starting argument values chosen by the planner for debugging and repair
- `steering_enabled`
  - indicates whether cross-conversation steering was active for the run
- `semantic_quality_flags`
  - stores known issues from validation for downstream inspection
- `used_repair`
  - indicates whether the deterministic repair pass was applied
- `repair_attempts`
  - records how many repair passes were attempted

Together these fields make it possible to inspect not just final quality, but also how and why a record was produced.

## 7. Validation And Repair

Validation is deterministic and covers:

- conversation structure
- tool-call endpoint existence in the registry
- required parameter presence
- grounding

The grounding validator was strengthened late in development. It now enforces entity-type-aware ID matching:

- `hotel_id` must use hotel IDs
- `flight_id` must use flight IDs
- `product_id` must use product IDs
- `restaurant_id` must use restaurant IDs

This closed an earlier weakness where any previously seen ID could satisfy grounding, even if it belonged to the wrong entity family.

Repair remains intentionally narrow:

- fill missing required params when possible
- replace hallucinated IDs only with compatible prior IDs
- add a final assistant response if missing

If repair cannot safely fix an incompatible ID, the record can remain invalid instead of being force-passed. That tradeoff favors honesty over masking errors.

## 8. Evaluation Design

Each conversation is scored on three dimensions:

- `naturalness`
- `tool_correctness`
- `task_completion`

These dimensions were chosen because they cover complementary parts of the problem:

- naturalness measures whether the conversation reads plausibly as a dialogue
- tool correctness measures whether tool selection, sequencing, and arguments are operationally appropriate
- task completion measures whether the user’s request appears to have been resolved

Together they provide a simple proxy for conversational quality, operational correctness, and outcome quality.

The current judge is heuristic-first. This choice was deliberate:

- it keeps the pipeline offline and deterministic
- it supports testability and reproducibility
- it avoids depending on an external model during submission runs

The judge also now penalizes:

- validation failure
- incompatible or unknown IDs
- cases where the user intent sounds search-oriented but the conversation does not begin with a search-like tool step

Aggregate metrics include:

- mean scores
- validation pass rate
- average message count
- average tool-call count
- tool usage entropy
- distinct tool-pair ratio
- category coverage

## Model Choices And Tradeoffs

The final system is offline and mostly rule/template-based for generation. The evaluator is heuristic-first rather than a full online LLM judge, and no external LLM dependency is required for the main submission pipeline.

This was chosen for:

- determinism
- reproducibility
- testability
- cost control

The tradeoff is straightforward:

- the system is easier to run, compare, and debug
- but semantic richness is weaker than a strong real LLM-based generator and judge would likely provide

Future work could keep the current architecture while swapping in stronger LLM-backed planning or judging components behind the same interfaces.

## 9. Cross-Conversation Steering And Diversity

Cross-conversation steering is lightweight and frequency-based.

The corpus state tracks:

- tool usage
- category usage
- tool-pair usage
- chain-length usage

When steering is enabled, lower-frequency tools and paths are preferred, but the seeded behavior remains deterministic. There is no vector memory, retrieval layer, or external memory system in this repository.

Run A vs Run B:

- Run A disables steering
- Run B enables steering

This supports a simple diversity experiment without changing the architecture.

The current steering approach also has clear limits:

- frequency-based steering improves spread but does not fully understand semantic intent
- small sample sizes make steering comparisons weak
- steering can diversify tool usage without guaranteeing more interesting conversational variety
- at larger scale, stronger memory/state or learned diversity control would likely help

## 10. Experimental Results

Observed results from saved repository outputs:

### Rich 100-sample run

From `data/outputs/final_report.json`:

- `num_records = 100`
- `mean_naturalness = 4.07`
- `mean_tool_correctness = 4.96`
- `mean_task_completion = 4.98`
- `mean_overall_score = 4.67`
- `validation_pass_rate = 0.96`
- `tool_usage_entropy = 3.298`
- `distinct_tool_pair_ratio = 0.3`

The end-to-end test asserts a minimum `mean_overall_score` threshold of `4.0`. That threshold was chosen because the current judge is heuristic-first and rewards structural correctness and task completion more than perfect semantic realism.

### Quality-check 20-sample run

From `data/outputs/quality_fix_check_report.json`:

- `num_records = 20`
- `mean_overall_score = 4.667`
- `validation_pass_rate = 0.9`
- `tool_usage_entropy = 3.218`
- `distinct_tool_pair_ratio = 0.433`

### Steering sample reports

From the saved small run reports:

- `run_a_report.json`
  - `5` records
  - `mean_overall_score = 4.667`
- `run_b_report.json`
  - `5` records
  - `mean_overall_score = 4.667`

These small steering comparison files are useful for reproducibility examples, but they are too small to claim a meaningful steering effect.

Qualitatively, the final tightening pass improved graph locality and chain realism by removing many unrelated cross-domain edges and by pushing the sampler toward search-led, same-family chains.

## Diversity & Quality Analysis

### Metrics Chosen And Why

The project tracks both diversity and quality-oriented metrics.

- `tool_usage_entropy`
  - measures how spread out tool usage is across a run
- `distinct_tool_pair_ratio`
  - measures compositional diversity by checking how many unique tool transitions appear relative to the total number of transitions
- `category_coverage`
  - gives a coarse view of how much category spread appears in the generated set
- `mean_overall_score`
  - summarizes the judge’s view of per-record quality
- `validation_pass_rate`
  - measures structural and grounding robustness

These metrics matter because entropy captures usage spread, tool-pair ratio captures chain composition diversity, and validation/mean scores capture whether the generator is still producing coherent and operationally sound records.

### Numeric Run A / Run B Results

The repository includes small saved steering comparison reports:

- Run A (`run_a_report.json`)
  - `num_records = 5`
  - `mean_overall_score = 4.667`
  - `tool_usage_entropy = 0.0`
  - `distinct_tool_pair_ratio = 0.0`
- Run B (`run_b_report.json`)
  - `num_records = 5`
  - `mean_overall_score = 4.667`
  - `tool_usage_entropy = 0.0`
  - `distinct_tool_pair_ratio = 0.0`

### Tradeoff Analysis

The current saved Run A / Run B sample is too small to support a strong claim about steering effectiveness. In these small saved runs, quality is effectively unchanged and diversity metrics do not separate meaningfully.

That does not mean steering is useless. It means the current stored comparison is not large enough to show the effect clearly. The richer quality-fixed run is more useful for qualitative inspection of the final pipeline, but it is not a substitute for a larger controlled steering experiment.

## 11. Failure Analysis

Important earlier failure modes included:

- weak grounding:
  - any prior ID could be reused even if it belonged to the wrong entity type
- overly permissive graph:
  - weak same-category edges created semantically odd transitions
- weak sampling locality:
  - chains could start sensibly and then drift into unrelated domains
- generic repair:
  - fallback IDs could be too generic or semantically wrong
- stale processed artifact mismatch:
  - a later build on `sample_tools.json` could overwrite default processed artifacts used by a rich run

What was fixed:

- grounding is now entity-type-aware
- graph scoring was tightened
- sampling prefers same-family and compatible-family paths
- repair only substitutes compatible IDs
- build now writes input-specific processed artifacts such as `sample_tools_rich_registry.json` and `sample_tools_rich_graph.json`
- generation can explicitly consume artifact paths via `--registry-path` and `--graph-path`

What still remains imperfect:

- some multi-step chains are structurally valid but still not fully natural
- travel-compatible paths like hotel <-> flight are intentionally allowed, and some may still feel weakly motivated
- the heuristic judge is useful but not equivalent to a strong model-based evaluator

## 12. Tradeoffs And Future Work

Current tradeoffs:

- offline determinism was prioritized over realism
- modularity and testability were prioritized over aggressive optimization
- repair is targeted and conservative rather than ambitious

Reasonable future work:

- stronger semantic validators beyond ID-family matching
- richer chain planning based on explicit task templates
- stricter graph pruning or learned transition scoring
- a stronger LLM-backed judge
- more realistic mock execution semantics
- richer repair that can revise plans instead of only patching local arguments

## 13. Reproducibility Appendix

### Main rich-sample pipeline

```powershell
python -m pip install -e .
python -m toolgen.cli build --input data/raw/sample_tools_rich.json
python -m toolgen.cli generate --num-samples 100 --seed 42 --output data/outputs/final_run.jsonl --registry-path data/processed/sample_tools_rich_registry.json --graph-path data/processed/sample_tools_rich_graph.json
python -m toolgen.cli evaluate --input data/outputs/final_run.jsonl --report-out data/outputs/final_report.json
python -m pytest -q
```

### Run A: steering disabled

```powershell
python -m toolgen.cli generate --num-samples 100 --seed 42 --output data/outputs/run_a.jsonl --registry-path data/processed/sample_tools_rich_registry.json --graph-path data/processed/sample_tools_rich_graph.json --no-cross-conversation-steering
python -m toolgen.cli evaluate --input data/outputs/run_a.jsonl --report-out data/outputs/run_a_report.json
```

### Run B: steering enabled

```powershell
python -m toolgen.cli generate --num-samples 100 --seed 42 --output data/outputs/run_b.jsonl --registry-path data/processed/sample_tools_rich_registry.json --graph-path data/processed/sample_tools_rich_graph.json
python -m toolgen.cli evaluate --input data/outputs/run_b.jsonl --report-out data/outputs/run_b_report.json
```

Key reproducibility points:

- use the rich raw sample
- use fixed seeds
- use the input-specific processed artifacts
- run evaluation on the same generated JSONL file that was just produced
