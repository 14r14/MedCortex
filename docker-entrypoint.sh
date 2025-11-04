#!/bin/sh
# Entrypoint script for Streamlit on IBM Code Engine
# Handles PORT environment variable that Code Engine may set

# Use PORT if set (Code Engine sets this), otherwise use default 8080
export STREAMLIT_SERVER_PORT=${PORT:-8080}

# Execute the main command
exec "$@"

