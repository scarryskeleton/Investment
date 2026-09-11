"""Streamlit rendering for the four dashboard modes.

app.py stays the entry point (page config, sidebar, mode dispatch, and the
not-yet-split Analyze mode); each mode's render function and its private
helpers live in their own module here to keep any one file from growing
without bound.
"""
