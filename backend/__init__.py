"""
backend
───────
FastAPI backend for the DynaMate2 React UI. Wraps the `dynamate` package
exactly as `app.py` (the Gradio UI) does — same PersistentAgentPoolWithSupervisor
construction order, same ui_state/ on-disk layout — behind an HTTP API instead
of Gradio Blocks.
"""
