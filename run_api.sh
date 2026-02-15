#!/bin/bash
cd "$(dirname "$0")"
exec python -m negotiation_chatbot.main < /dev/null
