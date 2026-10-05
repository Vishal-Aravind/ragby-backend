"""Website Flows: the visual bot that runs inside the embeddable chat widget.

Separate from the WhatsApp engine (flows.py): website flows are their own
flows (flows.channel = 'web'), keep their state in web_flow_sessions, and
answer the widget with render instructions instead of sending messages.
"""
