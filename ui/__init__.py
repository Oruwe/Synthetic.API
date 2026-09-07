# Makes `ui` importable as a package -- needed only by
# deploy/huggingface/merged_app.py, which imports `ui.app`'s Gradio
# `demo` object directly rather than launching it as a standalone script.
# The docker-compose deployment is unaffected: ui/Dockerfile still runs
# `python app.py` directly inside its own container, exactly as before --
# this file's mere presence changes nothing about that path.
