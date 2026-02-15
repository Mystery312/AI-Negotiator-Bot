End-to-End Pipeline (Pseudocode)

1. Startup
    load_dotenv()
    start FastAPI app (API, port 8001)
    start Gradio app (UI, port 7860)

    FastAPI startup_event():
        if PRELOAD_RAG == "true":
            preload_casino_rag()

2. Gradio UI + model discovery
    API_BASE_URL = first base in
        [env(API_BASE_URL),
         "http://api:8001",
         "http://localhost:8001",
         "http://127.0.0.1:8001",
         "http://localhost:8000"]
        that answers GET /health

    providers = llm_client.get_available_providers()
    all_models = flatten providers → "ollama:...", "gemini:..."

    UI holds:
        history_state (list of turns)
        item_names_state ({"item0": name, ...})
        item_counts_state ({"item0": int, ...})
        conversation_id, chosen_model, speaker names

3. Configure items
    configure_items(name0, name1, name2, count0, count1, count2):
        if any name empty or any count <= 0:
            clear item_*_state, show error
        else:
            item_names_state  = {"item0": name0, "item1": name1, "item2": name2}
            item_counts_state = {"item0": int(count0), "item1": int(count1), "item2": int(count2)}

4. User sends a turn (UI → API)
    on_send(role_radio, text, model, conv_id, history_state, speaker_a, speaker_b, item_names, item_counts):
        if text blank:
            return history_state unchanged

        if item_names or item_counts missing:
            append system error message, return

        if role_radio == "You":
            msg_role = "A"; backend_speaker = "A"; display_name = speaker_a or "You"
        else:
            msg_role = "B"; backend_speaker = "B"; display_name = speaker_b or "The other party"

        new_msg = {
            "role": msg_role,
            "speaker": backend_speaker,
            "display_name": display_name,
            "text": text,
            "move": None,
            "pd": None,
            "ts": now()
        }
        history_state.append(new_msg)
        save_conversation(conv_id, history_state)

        coach_advice, rag_source, rag_context = get_coach_advice(
            conv_id,
            msg_role,
            model,
            text=text,
            item_names=item_names,
            item_counts=item_counts
        )

        if coach_advice looks OK:
            history_state.append({
                "role": "Coach",
                "speaker": "Coach",
                "text": coach_advice,
                "move": None,
                "pd": None,
                "ts": now()
            })
            save_conversation(conv_id, history_state)

        if msg_role == "B":
            turns_text = [m["text"] for m in history_state if m["role"] in ["A", "B"]]
            proposal = autoplay.generate_bot_proposal(turns_text, item_counts)
            if proposal:
                proposal_text = autoplay.format_proposal_message(proposal, item_counts)
                history_state.append({
                    "role": "Coach",
                    "speaker": "Bot",
                    "text": proposal_text,
                    "move": None,
                    "pd": None,
                    "ts": now()
                })
                save_conversation(conv_id, history_state)

        next_role = "Other Party" if role_radio == "You" else "You"
        return updated history_state, rendered_chat, next_role

5. API: /chat (label → graph → coach → RAG log)
    POST /chat(ChatMessage msg):
        labels = label_text(msg.text)
        if "move" missing: labels["move"] = labels.get("move_type", "info_share")
        if "pd" missing:   labels["pd"]   = "C"

        try:
            upsert_turn(
                conv_id = msg.conv_id,
                speaker = msg.speaker,
                text    = msg.text,
                move    = labels["move"],
                pd      = labels["pd"]
            )
        except:
            continue

        if msg.provider given:
            provider = msg.provider
            model    = msg.model
        else:
            provider = "gemini" if msg.model startswith "gemini" else "ollama"
            model    = msg.model

        item_names  = msg.item_names  or {}
        item_counts = msg.item_counts or {"item0": 3, "item1": 2, "item2": 1}

        advice_result = get_advice_async(
            conv_id    = msg.conv_id,
            speaker    = msg.speaker,
            model      = model,
            provider   = provider,
            item_names = item_names,
            item_counts= item_counts
        )

        try:
            rag_src  = advice_result.get("rag_source", "none")
            rag_used = rag_src in ("casino", "generic")
            upsert_rag_usage(msg.conv_id, msg.speaker, rag_used)
        except:
            pass

        return {
            "advice":      advice_result["advice"],
            "reply":       advice_result["reply"],
            "rag_source":  advice_result.get("rag_source", "none"),
            "rag_context": advice_result.get("rag_context", "")
        }

6. Coach (get_advice_async)
    get_advice_async(conv_id, speaker, model, provider, item_names, item_counts):
        turns = fetch_last_n(conv_id, N)

        compute negotiation features:
            move types, pressure/relationship signals,
            numeric offers + concession history,
            deal / no-deal / Nash-like hints

        if preference model loaded:
            my_w, opp_w = estimate_preferences_cached(tuple(t["text"] for t in turns))
            candidate_best_offer = pareto.best_offer(my_w, opp_w, item_counts, ...)

        priorities = analyze_item_priorities_cached(
            turns_json    = json.dumps(turns),
            model         = model,
            item_names_json = json.dumps(item_names or None)
        )
        current_offers = extract_current_offers_cached(
            turns_json    = json.dumps(turns),
            model         = model,
            item_names_json = json.dumps(item_names or None)
        )

        rag_context, rag_source = retrieve_rag_context_cached(
            hint      = short text about what advice is needed,
            turns_json= json.dumps(turns)
        )

        system_prompt = "You are an AI negotiation coach..."
        user_context  = summary of:
            last turns, features, priorities, offers,
            any Pareto suggestion, rag_context

        llm_client = create_llm_client(provider, model)
        llm_answer = llm_client.generate_response(
            [{"role": "system", "content": system_prompt},
             {"role": "user",   "content": user_context}],
            temperature=0.4
        )

        advice_text    = postprocess_to_concise_advice(llm_answer)
        suggested_reply= optional_next_turn(llm_answer)

        return {
            "advice":      advice_text,
            "reply":       suggested_reply,
            "rag_source":  rag_source,
            "rag_context": rag_context or ""
        }

7. RAG (generic tactics)
    retrieve_rag_context(query, n_results=5):
        model, collection = _ensure_initialized()
            if first time:
                embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                client          = chromadb.PersistentClient("./chroma_db")
                collection      = client.get_or_create_collection("negotiation_tactics")
                if empty: add_sample_data_to_collection(collection, embedding_model)

        q_emb   = embedding_model.encode(query)
        results = collection.query(query_embeddings=[q_emb], n_results=n_results)
        docs    = results["documents"][0]
        if not docs: return ""
        return "\n\n\n\n".join(docs)

8. LLM client abstraction
    create_llm_client(provider, model_name):
        if provider == "ollama": return OllamaProvider(model_name)
        if provider == "gemini": return GeminiProvider(model_name)

    OllamaProvider.generate_response(messages, **kw):
        client   = OpenAI(api_key="dummy", base_url=OLLAMA_BASE_URL + "/v1")
        response = client.chat.completions.create(model=self.model_name, messages=messages, **kw)
        return response.choices[0].message.content

    GeminiProvider.generate_response(messages, **kw):
        convert messages → Gemini format
        response = self.model.generate_content(converted_messages)
        return response.text

9. Graph + stats + visualizations
    API:
        GET /graph/{conv_id} → get_conversation_graph_data(conv_id)
        GET /stats/{conv_id} → get_conversation_stats(conv_id)

    UI:
        graph_data = GET /graph/{conv_id}
        stats      = GET /stats/{conv_id}
        graph_plot = create_graph_visualization(graph_data)
        stats_plot = create_conversation_stats(conv_id, stats)

10. DoND visualizer + Pareto simulator
    load_dond_sample_viz(...):
        load VAL_SAMPLES from deal_or_no_dialog/exported/validation.jsonl
        choose sample (optionally only no-deal)
        build:
            item_counts markdown,
            speaker_stats markdown,
            message_timeline table,
            speaker_plot, content_plot
        optionally call get_coach_advice() inline per turn

    run_pareto_sim(n, baseline, ratio, model):
        summary = simulate_dond.simulate_with_coach(n, baseline, ratio)
        build markdown summary
        for each transcript:
            optionally replay to get_coach_advice() and embed last advice line

11. OpenAI-compatible endpoint
    POST /v1/chat/completions(ChatCompletionReq req):
        user_msg = first user message
        text     = user_msg.content
        conv_id  = req.model
        speaker  = "User"

        labels = label_text(text)
        ensure labels["move"], labels["pd"]
        upsert_turn(conv_id, speaker, text, labels["move"], labels["pd"])

        result    = get_advice(conv_id, speaker)
        reply_msg = Message(role="assistant", content=result["reply"])

        return ChatCompletionResp(
            id      = "chatcmpl-<uuid>",
            choices = [Choice(message = reply_msg)],
            created = now(),
            model   = req.model
        )